import subprocess, json, tempfile, os, sys, random
from concurrent.futures import ProcessPoolExecutor
PY_="/Users/alif_1/Documents/battlebots/.venv-bot/bin/python"; REC="tools/record_reference_match.py"
# Robust design: marginalize out field composition. Every game contains the
# CANDIDATE and v12 together (paired, identical game). The other 6 seats are
# drawn deterministically-by-seed from a broad pool. Report the paired result:
# how often candidate out-ranks v12 in the SAME game, + mean rank/mass each.
CAND=sys.argv[2] if len(sys.argv)>2 else "dist/potential_field_v13h.py"
CNAME=sys.argv[3] if len(sys.argv)>3 else "v13h"
POOL=[("v6","dist/potential_field_v6.py"),("v8","dist/potential_field_v8.py"),
      ("v9","dist/potential_field_v9.py"),("v10","dist/potential_field_v10.py"),
      ("v11","dist/potential_field_v11.py"),("aggro","dist/spar_aggro.py"),
      ("v13c","dist/potential_field_v13c.py"),("v13e","dist/potential_field_v13e.py"),
      ("v11a","dist/potential_field_v11a.py"),("v11abg","dist/potential_field_v11abg.py")]
def run_one(seed):
    rng=random.Random(seed*7919+13)
    opps=rng.sample(POOL,6)
    field=[("v12","dist/potential_field_v12.py"),(CNAME,CAND)]+opps
    rng.shuffle(field)  # randomize seats too
    specs=[f"1:{f[1]}" for f in field]
    fd,tmp=tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY_,REC,"--seed",str(seed),"--rounds","1400","--out",tmp,"--bots",*specs],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    t=json.loads(open(tmp).read()); os.unlink(tmp)
    fm={int(k):v for k,v in t["final_masses"].items()}
    rank={p:i+1 for i,p in enumerate(t["rankings"])}
    cand_seat=[i for i,f in enumerate(field) if f[0]==CNAME][0]
    v12_seat=[i for i,f in enumerate(field) if f[0]=="v12"][0]
    return (rank[cand_seat], fm[cand_seat], rank[v12_seat], fm[v12_seat])
if __name__=="__main__":
    N=int(sys.argv[1]) if len(sys.argv)>1 else 96
    with ProcessPoolExecutor(max_workers=8) as ex:
        res=list(ex.map(run_one, range(N)))
    cr=[r[0] for r in res]; cm=[r[1] for r in res]
    vr=[r[2] for r in res]; vm=[r[3] for r in res]
    cand_better=sum(1 for r in res if r[0]<r[2])   # lower rank = better
    v12_better =sum(1 for r in res if r[2]<r[0])
    ties=sum(1 for r in res if r[0]==r[2])
    print(f"ENSEMBLE {N} games (each = {CNAME} + v12 + 6 random pool opps, paired):")
    print(f"  {CNAME}: mean rank {sum(cr)/N:.3f}  mean mass {sum(cm)/N:.3f}")
    print(f"  v12  : mean rank {sum(vr)/N:.3f}  mean mass {sum(vm)/N:.3f}")
    print(f"  PAIRED: {CNAME} out-ranks v12 in {cand_better}/{N}  | v12 better {v12_better}/{N} | ties {ties}")
    print(f"  mean rank delta ({CNAME}-v12) = {(sum(cr)-sum(vr))/N:+.3f}  (negative = {CNAME} better)")
