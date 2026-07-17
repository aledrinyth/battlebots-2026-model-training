import subprocess, json, tempfile, os, sys
from concurrent.futures import ProcessPoolExecutor
PY_="/Users/alif_1/Documents/battlebots/.venv-bot/bin/python"; REC="tools/record_reference_match.py"
# Focused field: baselines (v8,v10,v11,v12 + v12 anchor) vs standoff candidates.
# All 8 share the same games => inherently paired per seed/seat.
FIELD=[("v8","dist/potential_field_v8.py"),
       ("v10","dist/potential_field_v10.py"),
       ("v11","dist/potential_field_v11.py"),
       ("v12","dist/potential_field_v12.py"),
       ("v12b","dist/potential_field_v12.py"),   # anchor (tie calibration)
       ("v13f","dist/potential_field_v13f.py"),
       ("v13g","dist/potential_field_v13g.py"),
       ("v13h","dist/potential_field_v13h.py")]
def run_one(args):
    seed, shift = args
    order=FIELD[shift:]+FIELD[:shift]
    specs=[f"1:{f[1]}" for f in order]
    fd,tmp=tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY_,REC,"--seed",str(seed),"--rounds","1400","--out",tmp,"--bots",*specs],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    t=json.loads(open(tmp).read()); os.unlink(tmp)
    fm={int(k):v for k,v in t["final_masses"].items()}
    rank={p:i+1 for i,p in enumerate(t["rankings"])}
    out={}
    for seat,f in enumerate(order):
        nm="v12" if f[0]=="v12b" else f[0]
        d=out.setdefault(nm,{"mass":0.0,"rank":0.0,"n":0,"win":0})
        d["mass"]+=fm[seat]; d["rank"]+=rank[seat]; d["n"]+=1
        d["win"]+= 1 if rank[seat]==1 else 0
    return out
if __name__=="__main__":
    N=int(sys.argv[1]) if len(sys.argv)>1 else 48
    jobs=[(s, s%8) for s in range(N)]
    agg={}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for out in ex.map(run_one, jobs):
            for nm,d in out.items():
                a=agg.setdefault(nm,{"mass":0.0,"rank":0.0,"n":0,"win":0})
                a["mass"]+=d["mass"]; a["rank"]+=d["rank"]; a["n"]+=d["n"]; a["win"]+=d["win"]
    print(f"FOCUSED MULTIWAY {N} seeds (8 distinct bots/game, seat-rotated, paired):")
    print(f"{'type':6} {'seatgames':>9} {'meanMass':>9} {'meanRank':>9} {'wins':>5}")
    for nm in sorted(agg, key=lambda k: agg[k]['rank']/agg[k]['n']):
        a=agg[nm]; print(f"{nm:6} {a['n']:9d} {a['mass']/a['n']:9.3f} {a['rank']/a['n']:9.3f} {a['win']:5d}")
