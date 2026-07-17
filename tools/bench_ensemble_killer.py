import subprocess, json, tempfile, os, sys, random
from concurrent.futures import ProcessPoolExecutor
PY_="/Users/alif_1/Documents/battlebots/.venv-bot/bin/python"; REC="tools/record_reference_match.py"
# DECISIVE test: every game contains v12 + candidate + 2 aggressive split-lunge
# KILLERS (tournament-winner proxy) + 4 random foragers. Paired per game.
# Measures whether the candidate's extended split-standoff actually helps SURVIVE
# the split-lunge threat the self-play pool otherwise lacks. Tracks rank + deaths.
CAND=sys.argv[2] if len(sys.argv)>2 else "dist/potential_field_v13h.py"
CNAME=sys.argv[3] if len(sys.argv)>3 else "v13h"
FORAGE=[("v6","dist/potential_field_v6.py"),("v8","dist/potential_field_v8.py"),
        ("v9","dist/potential_field_v9.py"),("v10","dist/potential_field_v10.py"),
        ("v11","dist/potential_field_v11.py"),("v11a","dist/potential_field_v11a.py")]
def deaths_of(t, seat):
    d=0; alive=True
    for rd in t["rounds"]:
        snap=rd["snaps"].get(str(seat))
        if snap is None: continue
        al=snap.get("alive")
        if alive and not al: d+=1
        alive=al
    return d
def run_one(seed):
    rng=random.Random(seed*7919+13)
    opps=rng.sample(FORAGE,4)
    field=[("v12","dist/potential_field_v12.py"),(CNAME,CAND),
           ("k","dist/spar_killer.py"),("k","dist/spar_killer.py")]+opps
    rng.shuffle(field)
    specs=[f"1:{f[1]}" for f in field]
    fd,tmp=tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY_,REC,"--seed",str(seed),"--rounds","1400","--out",tmp,"--bots",*specs],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    t=json.loads(open(tmp).read()); os.unlink(tmp)
    fm={int(k):v for k,v in t["final_masses"].items()}
    rank={p:i+1 for i,p in enumerate(t["rankings"])}
    cs=[i for i,f in enumerate(field) if f[0]==CNAME][0]
    vs=[i for i,f in enumerate(field) if f[0]=="v12"][0]
    return (rank[cs],fm[cs],deaths_of(t,cs), rank[vs],fm[vs],deaths_of(t,vs))
if __name__=="__main__":
    N=int(sys.argv[1]) if len(sys.argv)>1 else 96
    with ProcessPoolExecutor(max_workers=8) as ex:
        res=list(ex.map(run_one, range(N)))
    cr=[r[0] for r in res]; cm=[r[1] for r in res]; cd=[r[2] for r in res]
    vr=[r[3] for r in res]; vm=[r[4] for r in res]; vd=[r[5] for r in res]
    cb=sum(1 for r in res if r[0]<r[3]); vb=sum(1 for r in res if r[3]<r[0])
    print(f"KILLER-ENSEMBLE {N} games ({CNAME}+v12+2 killers+4 foragers, paired):")
    print(f"  {CNAME}: mean rank {sum(cr)/N:.3f}  mass {sum(cm)/N:.3f}  deaths {sum(cd)/N:.3f}")
    print(f"  v12  : mean rank {sum(vr)/N:.3f}  mass {sum(vm)/N:.3f}  deaths {sum(vd)/N:.3f}")
    print(f"  PAIRED rank: {CNAME} better {cb}/{N} | v12 better {vb}/{N} | ties {N-cb-vb}")
    print(f"  rank delta ({CNAME}-v12) {(sum(cr)-sum(vr))/N:+.3f} | deaths delta {(sum(cd)-sum(vd))/N:+.3f} (neg=fewer deaths)")
