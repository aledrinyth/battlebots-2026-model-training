(module
  (memory (export "mem") 2)

  ;; engine-exact per-turn speed: max(0.25, 1.1/(1+0.08 r))
  (func $speed (param $r f64) (result f64)
    (local $v f64)
    (local.set $v
      (f64.div (f64.const 1.1)
        (f64.add (f64.const 1.0)
          (f64.mul (f64.const 0.08) (local.get $r)))))
    (if (result f64) (f64.lt (local.get $v) (f64.const 0.25))
      (then (f64.const 0.25))
      (else (local.get $v))))

  ;; score_candidates: evaluate every candidate heading one frame ahead under
  ;; the effective-kill-radius opponent model; write per-candidate value to the
  ;; output block and return the best candidate index (i32).
  ;;
  ;; Memory layout (bytes):
  ;;   0:i32 n_ours   4:i32 n_en   8:i32 n_dirs
  ;;   16:f64 ratio  24:f64 death_pen  32:f64 gain_w  40:f64 align_w
  ;;   48:f64 split_reach  56:f64 split_min_mass  64:f64 arena
  ;;   72:f64 base_ux  80:f64 base_uy
  ;;   128: our blobs  [x,y,r] x n_ours   (24 bytes each)
  ;;   then enemies    [x,y,r] x n_en
  ;;   then dirs       [ux,uy] x n_dirs   (16 bytes each)
  ;;   65536: output   [value] x n_dirs   (8 bytes each)
  (func (export "score_candidates") (result i32)
    (local $n_ours i32) (local $n_en i32) (local $n_dirs i32)
    (local $ratio f64) (local $death f64) (local $gainw f64) (local $alignw f64)
    (local $sreach f64) (local $smin f64) (local $arena f64)
    (local $bux f64) (local $buy f64)
    (local $ourbase i32) (local $enbase i32) (local $dirbase i32)
    (local $d i32) (local $i i32) (local $j i32)
    (local $ux f64) (local $uy f64)
    (local $surv f64) (local $lost f64) (local $gained f64)
    (local $ox f64) (local $oy f64) (local $orr f64) (local $mi f64)
    (local $sp f64) (local $nx f64) (local $ny f64)
    (local $ex f64) (local $ey f64) (local $er f64) (local $em f64)
    (local $killr f64) (local $ddx f64) (local $ddy f64) (local $d2 f64)
    (local $dead i32) (local $val f64) (local $lo f64) (local $hi f64)
    (local $best_idx i32) (local $best_val f64)
    (local $oaddr i32) (local $eaddr i32)

    (local.set $n_ours (i32.load (i32.const 0)))
    (local.set $n_en   (i32.load (i32.const 4)))
    (local.set $n_dirs (i32.load (i32.const 8)))
    (local.set $ratio  (f64.load (i32.const 16)))
    (local.set $death  (f64.load (i32.const 24)))
    (local.set $gainw  (f64.load (i32.const 32)))
    (local.set $alignw (f64.load (i32.const 40)))
    (local.set $sreach (f64.load (i32.const 48)))
    (local.set $smin   (f64.load (i32.const 56)))
    (local.set $arena  (f64.load (i32.const 64)))
    (local.set $bux    (f64.load (i32.const 72)))
    (local.set $buy    (f64.load (i32.const 80)))

    (local.set $ourbase (i32.const 128))
    (local.set $enbase (i32.add (i32.const 128)
      (i32.mul (local.get $n_ours) (i32.const 24))))
    (local.set $dirbase (i32.add (local.get $enbase)
      (i32.mul (local.get $n_en) (i32.const 24))))

    (local.set $best_idx (i32.const -1))
    (local.set $best_val (f64.const -1e300))

    (local.set $d (i32.const 0))
    (block $dbreak
    (loop $dloop
      (br_if $dbreak (i32.ge_s (local.get $d) (local.get $n_dirs)))
      (local.set $ux (f64.load (i32.add (local.get $dirbase)
        (i32.mul (local.get $d) (i32.const 16)))))
      (local.set $uy (f64.load (i32.add (local.get $dirbase)
        (i32.add (i32.mul (local.get $d) (i32.const 16)) (i32.const 8)))))
      (local.set $surv (f64.const 0))
      (local.set $lost (f64.const 0))
      (local.set $gained (f64.const 0))

      ;; loop over our blobs
      (local.set $i (i32.const 0))
      (block $ibreak
      (loop $iloop
        (br_if $ibreak (i32.ge_s (local.get $i) (local.get $n_ours)))
        (local.set $oaddr (i32.add (local.get $ourbase)
          (i32.mul (local.get $i) (i32.const 24))))
        (local.set $ox (f64.load (local.get $oaddr)))
        (local.set $oy (f64.load (i32.add (local.get $oaddr) (i32.const 8))))
        (local.set $orr (f64.load (i32.add (local.get $oaddr) (i32.const 16))))
        (local.set $mi (f64.mul (local.get $orr) (local.get $orr)))
        (local.set $sp (call $speed (local.get $orr)))
        ;; nx = clamp(ox+ux*sp, orr, arena-orr)
        (local.set $lo (local.get $orr))
        (local.set $hi (f64.sub (local.get $arena) (local.get $orr)))
        (local.set $nx (f64.add (local.get $ox) (f64.mul (local.get $ux) (local.get $sp))))
        (local.set $nx (f64.min (f64.max (local.get $nx) (local.get $lo)) (local.get $hi)))
        (local.set $ny (f64.add (local.get $oy) (f64.mul (local.get $uy) (local.get $sp))))
        (local.set $ny (f64.min (f64.max (local.get $ny) (local.get $lo)) (local.get $hi)))
        (local.set $dead (i32.const 0))

        ;; loop over enemies
        (local.set $j (i32.const 0))
        (block $jbreak
        (loop $jloop
          (br_if $jbreak (i32.ge_s (local.get $j) (local.get $n_en)))
          (local.set $eaddr (i32.add (local.get $enbase)
            (i32.mul (local.get $j) (i32.const 24))))
          (local.set $ex (f64.load (local.get $eaddr)))
          (local.set $ey (f64.load (i32.add (local.get $eaddr) (i32.const 8))))
          (local.set $er (f64.load (i32.add (local.get $eaddr) (i32.const 16))))
          (local.set $em (f64.mul (local.get $er) (local.get $er)))
          (local.set $ddx (f64.sub (local.get $nx) (local.get $ex)))
          (local.set $ddy (f64.sub (local.get $ny) (local.get $ey)))
          (local.set $d2 (f64.add (f64.mul (local.get $ddx) (local.get $ddx))
                                  (f64.mul (local.get $ddy) (local.get $ddy))))
          ;; threat: em >= ratio*mi ?
          (if (f64.ge (local.get $em) (f64.mul (local.get $ratio) (local.get $mi)))
            (then
              ;; killr = er + speed(er)
              (local.set $killr (f64.add (local.get $er) (call $speed (local.get $er))))
              ;; split upgrade: em>=smin and em*0.5>=ratio*mi -> max(killr, er+sreach)
              (if (i32.and
                    (f64.ge (local.get $em) (local.get $smin))
                    (f64.ge (f64.mul (local.get $em) (f64.const 0.5))
                            (f64.mul (local.get $ratio) (local.get $mi))))
                (then
                  (local.set $killr (f64.max (local.get $killr)
                    (f64.add (local.get $er) (local.get $sreach))))))
              (if (f64.le (local.get $d2) (f64.mul (local.get $killr) (local.get $killr)))
                (then (local.set $dead (i32.const 1))))))
          ;; upside: mi >= ratio*em and d2 <= orr*orr -> gain em
          (if (i32.and
                (f64.ge (local.get $mi) (f64.mul (local.get $ratio) (local.get $em)))
                (f64.le (local.get $d2) (f64.mul (local.get $orr) (local.get $orr))))
            (then (local.set $gained (f64.add (local.get $gained) (local.get $em)))))
          (local.set $j (i32.add (local.get $j) (i32.const 1)))
          (br $jloop)))

        (if (i32.eq (local.get $dead) (i32.const 1))
          (then (local.set $lost (f64.add (local.get $lost) (local.get $mi))))
          (else (local.set $surv (f64.add (local.get $surv) (local.get $mi)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $iloop)))

      ;; val = surv + gainw*gained - death*lost + alignw*(ux*bux+uy*buy)
      (local.set $val
        (f64.add
          (f64.sub
            (f64.add (local.get $surv) (f64.mul (local.get $gainw) (local.get $gained)))
            (f64.mul (local.get $death) (local.get $lost)))
          (f64.mul (local.get $alignw)
            (f64.add (f64.mul (local.get $ux) (local.get $bux))
                     (f64.mul (local.get $uy) (local.get $buy))))))
      (f64.store (i32.add (i32.const 65536) (i32.mul (local.get $d) (i32.const 8)))
                 (local.get $val))
      (if (f64.gt (local.get $val) (local.get $best_val))
        (then
          (local.set $best_val (local.get $val))
          (local.set $best_idx (local.get $d))))
      (local.set $d (i32.add (local.get $d) (i32.const 1)))
      (br $dloop)))

    (local.get $best_idx))
)
