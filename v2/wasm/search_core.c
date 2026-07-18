/* search_core.c — C source of the v9 tactical kernel (production build path).
 *
 * Compiles to wasm32-wasi with wasi-sdk clang; the resulting module exports the
 * same ABI as the hand-written kernel.wat (a fixed linear-memory layout + a
 * score_candidates() entry), so it is a drop-in for the embedded kernel:
 *
 *   $WASI_SDK/bin/clang --target=wasm32-wasi -O3 -nostdlib \
 *       -Wl,--no-entry -Wl,--export=score_candidates -Wl,--export=mem \
 *       -Wl,--export=__heap_base -o kernel.wasm search_core.c
 *
 * then zlib+base85 the output and paste into _KERNEL_B85 (see build_kernel.py).
 *
 * This file exists for two reasons: (1) it documents the kernel's semantics in
 * a readable language (the WAT is the shipped artifact only because no wasm
 * toolchain is installed on the dev box); (2) it is the seam for the MULTI-FRAME
 * search that hand-written WAT is impractical for — extend score_candidates()
 * with an inner ply loop + a beam over MEM, keeping everything in one wasm call
 * so the Python<->wasm FFI crossing is paid once per turn, not per node.
 *
 * Layout (doubles unless noted), identical to kernel.wat:
 *   0:i32 n_ours  4:i32 n_en  8:i32 n_dirs
 *   16: ratio death gain_w align_w split_reach split_min arena base_ux base_uy
 *   128: ours[n_ours]{x,y,r}  then en[n_en]{x,y,r}  then dirs[n_dirs]{ux,uy}
 *   65536: out[n_dirs] value
 */

typedef unsigned int u32;

/* One flat 128 KiB memory region, exported as "mem". */
__attribute__((aligned(8))) unsigned char mem[131072];

static double dload(u32 off) { return *(double *)(mem + off); }
static void dstore(u32 off, double v) { *(double *)(mem + off) = v; }
static u32 iload(u32 off) { return *(u32 *)(mem + off); }

static double dspeed(double r) {
    double v = 1.1 / (1.0 + 0.08 * r);
    return v < 0.25 ? 0.25 : v;
}

__attribute__((export_name("score_candidates")))
int score_candidates(void) {
    u32 n_ours = iload(0), n_en = iload(4), n_dirs = iload(8);
    double ratio = dload(16), death = dload(24), gainw = dload(32),
           alignw = dload(40), sreach = dload(48), smin = dload(56),
           arena = dload(64), bux = dload(72), buy = dload(80);
    u32 ourbase = 128, enbase = 128 + n_ours * 24, dirbase = enbase + n_en * 24;

    int best_idx = -1;
    double best_val = -1e300;
    for (u32 d = 0; d < n_dirs; d++) {
        double ux = dload(dirbase + d * 16), uy = dload(dirbase + d * 16 + 8);
        double surv = 0, lost = 0, gained = 0;
        for (u32 i = 0; i < n_ours; i++) {
            u32 o = ourbase + i * 24;
            double ox = dload(o), oy = dload(o + 8), orr = dload(o + 16);
            double mi = orr * orr, sp = dspeed(orr), lo = orr, hi = arena - orr;
            double nx = ox + ux * sp, ny = oy + uy * sp;
            nx = nx < lo ? lo : (nx > hi ? hi : nx);
            ny = ny < lo ? lo : (ny > hi ? hi : ny);
            int dead = 0;
            for (u32 j = 0; j < n_en; j++) {
                u32 e = enbase + j * 24;
                double ex = dload(e), ey = dload(e + 8), er = dload(e + 16);
                double em = er * er, ddx = nx - ex, ddy = ny - ey;
                double d2 = ddx * ddx + ddy * ddy;
                if (em >= ratio * mi) {
                    double killr = er + dspeed(er);
                    if (em >= smin && em * 0.5 >= ratio * mi) {
                        double sk = er + sreach;
                        if (sk > killr) killr = sk;
                    }
                    if (d2 <= killr * killr) dead = 1;
                }
                if (mi >= ratio * em && d2 <= orr * orr) gained += em;
            }
            if (dead) lost += mi; else surv += mi;
        }
        double val = surv + gainw * gained - death * lost
                     + alignw * (ux * bux + uy * buy);
        dstore(65536 + d * 8, val);
        if (val > best_val) { best_val = val; best_idx = (int)d; }
    }
    return best_idx;
}
