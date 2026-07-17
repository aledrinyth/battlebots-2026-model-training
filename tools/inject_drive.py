import sys, json, tempfile, os, subprocess
from concurrent.futures import ThreadPoolExecutor
REPO="/Users/alif_1/Documents/battlebots"; PY=f"{REPO}/.venv-bot/bin/python"
SCRIPT=f"{REPO}/tools/inject_predator_match.py"
V12="dist/potential_field_v12.py"
CAND=sys.argv[3] if len(sys.argv)>3 else "dist/potential_field_v13h.py"
CNAME=sys.argv[4] if len(sys.argv)>4 else "v13h"
PMASS=sys.argv[5] if len(sys.argv)>5 else "40"
INJ="400"; WIN="150"
def job(args):
    seed, offset, swap = args
    t0, t1 = (V12, CAND) if not swap else (CAND, V12)
    fd,tmp=tempfile.mkstemp(suffix=".json"); os.close(fd)
    cmd=[PY, SCRIPT, "--seed",str(seed), "--inject-round",INJ, "--window",WIN,
         "--pred-mass",str(PMASS), "--offset",str(offset),
         "--target0",t0, "--target1",t1, "--out",tmp]
    try:
        subprocess.run(cmd, cwd=REPO, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r=json.loads(open(tmp).read())
    except Exception:
        return offset, None
    finally:
        try: os.unlink(tmp)
        except OSError: pass
    if not swap:
        return offset, {"v12_died":r["died0"],"cand_died":r["died1"],
                        "v12_mass":r["mass0"],"cand_mass":r["mass1"]}
    return offset, {"v12_died":r["died1"],"cand_died":r["died0"],
                    "v12_mass":r["mass1"],"cand_mass":r["mass0"]}
if __name__=="__main__":
    N=int(sys.argv[1]); offsets=[float(x) for x in sys.argv[2].split(",")]
    jobs=[(s, off, s%2==1) for off in offsets for s in range(N)]
    agg={off:{"v12_d":0,"c_d":0,"v12_m":0.0,"c_m":0.0,"n":0,"fail":0} for off in offsets}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for off,d in ex.map(job, jobs):
            a=agg[off]
            if d is None: a["fail"]+=1; continue
            a["n"]+=1
            a["v12_d"]+=d["v12_died"]; a["c_d"]+=d["cand_died"]
            a["v12_m"]+=d["v12_mass"]; a["c_m"]+=d["cand_mass"]
    print(f"INJECTION STRESS: predator mass {PMASS} at round {INJ}, window {WIN}, cand={CNAME}")
    print(f"  (paired same aged arena, seat-swapped to cancel bias)")
    print(f"{'offset':>7} {'N':>4} {'fail':>4} | {'v12 die%':>9} {CNAME+' die%':>12} | {'v12 mass':>9} {CNAME+' mass':>11}")
    for off in offsets:
        a=agg[off]; n=max(1,a["n"])
        print(f"{off:7.0f} {a['n']:4d} {a['fail']:4d} | {100*a['v12_d']/n:9.1f} {100*a['c_d']/n:12.1f} | "
              f"{a['v12_m']/n:9.2f} {a['c_m']/n:11.2f}")
