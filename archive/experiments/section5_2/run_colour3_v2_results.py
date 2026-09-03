# Section 5.3 colour three-node V2 results. Readout only.
from __future__ import annotations
import glob, os, statistics as st, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0,"experiments/section5_2"); sys.path.insert(0,"src"); sys.path.insert(0,"experiments")
from averaged_readout import load_run, metrics, averaged_params, truth_for_run, wrap_np
from beliefmesh.node.mesh import fov_cells
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged

G,FOV,LAST=22,7,50
A,B,C=(8,9),(13,9),(10,12)                 # A red anchor, B blue anchor, C student
a=set(fov_cells(*A,FOV,G)); b=set(fov_cells(*B,FOV,G)); c=set(fov_cells(*C,FOV,G))
SHARED=np.array(sorted(c&(a|b)))           # 28 cells
Cmask=np.zeros((G,G),bool)
for r,cc in c: Cmask[r,cc]=True
# covering node ids per shared cell, ascending -> slot order
cov={}
for r,cc in map(tuple,SHARED):
    ids=[i for i,s in enumerate((a,b,c)) if (r,cc) in s]
    cov[(r,cc)]=ids
ARMS=["frozen","naive","certainty","nig_product","nig_product_avgtrain"]
ROOT="runs/chapter5_v2/s5_3_colour_three_node_v2"

def one(d,seed):
    d=Path(d); R=load_run(d,seed); T=R["T"]
    m=metrics(R,last=LAST,mask=Cmask)["averaged"]
    bel=np.load(d/"cell_beliefs_steps.npy").astype(np.float64)
    nc=np.load(d/"cell_ncov_steps.npy").astype(int)
    gam,nu_s,al_s,be_s=averaged_params(bel,nc)
    epi=be_s/(np.maximum(nu_s,1e-6)*np.maximum(al_s-1.0,1e-6))
    cert=1.0/(1.0+np.minimum(epi,10.0))
    cm=np.where(Cmask[None],R["averaged"][0],np.nan); cc_=np.where(Cmask[None],cert,np.nan)
    r_cert,_,_,_=per_timestep_then_averaged(cm,cc_,T-LAST,T)
    tr=truth_for_run(d,seed,T,G)[-LAST:]
    sub=bel[-LAST:][:,SHARED[:,0],SHARED[:,1],:,:]     # (50, 28, slot, 4)
    nu=sub[...,1]; g=sub[...,0]
    err=np.abs(wrap_np(g-tr[:,SHARED[:,0],SHARED[:,1]][...,None]))*180
    fin=np.isfinite(nu)&np.isfinite(err)
    # anchors-only mask: slot j is an anchor iff cov[cell][j] in {0,1}
    anch=np.zeros(nu.shape,bool)
    for k,(r_,cc_2) in enumerate(map(tuple,SHARED)):
        for j,nid in enumerate(cov[(r_,cc_2)]):
            if nid in (0,1): anch[:,k,j]=True
    rall=float(np.corrcoef(nu[fin],err[fin])[0,1])
    ma=fin&anch
    ranch=float(np.corrcoef(nu[ma],err[ma])[0,1]) if ma.sum()>10 else float("nan")
    n=np.isfinite(g).sum(-1); ok=n>=2
    nucv=float(np.nanmean(np.where(ok,np.nanstd(np.where(np.isfinite(g),nu,np.nan),-1)/
              np.maximum(np.nanmean(np.where(np.isfinite(g),nu,np.nan),-1),1e-12),np.nan)))
    x=np.where(np.isfinite(g),wrap_np(g-g[...,0][...,None]),np.nan)
    gs=float(np.nanmean(np.where(ok,np.nanmax(x,-1)-np.nanmin(x,-1),np.nan)))*180
    return dict(**m,r_cert=r_cert,r_all=rall,r_anch=ranch,nucv=nucv,gs=gs)

D={}
for arm in ARMS:
    V=[]
    for p in sorted(glob.glob("%s/%s_seed*/manifest.yaml"%(ROOT,arm))):
        d=Path(os.path.dirname(p)); s=int(yaml.safe_load(open(p))["env_seed"])
        V.append(one(d,s))
    if V: D[arm]=V
ms=lambda V,k:(st.mean([v[k] for v in V]), st.stdev([v[k] for v in V]) if len(V)>1 else 0.0)
print("="*104)
print("V2 -- metrics over the STUDENT C's full 49-cell FOV, averaged readout, 5 seeds")
print("="*104)
print("%-24s %3s %-20s %-20s %-15s %-15s %-15s"
      % ("arm","n","whole-run MSE","last-50 MSE","90% coverage","hw : RMS","cert-MSE r"))
for arm in ARMS:
    if arm not in D: continue
    V=D[arm]; f=lambda k,p=5:"%.*f +/- %.*f"%(p,ms(V,k)[0],p,ms(V,k)[1])
    print("%-24s %3d %-20s %-20s %-15s %-15s %-15s"
          % (arm,len(V),f("whole"),f("last50"),f("cov",3),f("ratio",3),f("r_cert",3)))
print("\n"+"="*104)
print("THE DECIDING DIAGNOSTIC -- r(contributor nu, contributor error) on C's 28 shared cells")
print("SUCCESS = NEGATIVE.  v1 measured +0.18 to +0.23 (all contributors).")
print("="*104)
print("%-24s %-22s %-22s %-14s %-14s"
      % ("arm","r(nu,err) all contrib","r(nu,err) anchors only","nu-CV","gamma spread"))
for arm in ARMS:
    if arm not in D: continue
    V=D[arm]; f=lambda k,p=3:"%+.*f +/- %.*f"%(p,ms(V,k)[0],p,ms(V,k)[1])
    print("%-24s %-22s %-22s %-14s %-14s"
          % (arm,f("r_all"),f("r_anch"),"%.3f"%ms(V,"nucv")[0],"%.2f deg"%ms(V,"gs")[0]))
print("\n  v1 reference (last-50): nu-CV 0.550-0.620, gamma spread 20.7-20.9 deg")
print("  offset-world reference: nu-CV 0.101, gamma spread 2-5 deg")
