# Section 5.3 three-node colour-world results. Readout only.
from __future__ import annotations
import glob, os, statistics as st, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0,"experiments/section5_2"); sys.path.insert(0,"src")
sys.path.insert(0,"experiments")
from averaged_readout import load_run, metrics, averaged_params, truth_for_run, wrap_np
from beliefmesh.node.mesh import fov_cells
from stage6_spatial_mesh.cert_mse_metrics import per_timestep_then_averaged

G,FOV,LAST=22,7,50
A,C,B=(8,10),(12,10),(10,13)
ARMS=["frozen","naive","certainty","nig_product","nig_product_avgtrain"]
ROOT="runs/chapter5_v2/s5_3_colour_three_node"
a=set(fov_cells(*A,FOV,G)); c=set(fov_cells(*C,FOV,G)); b=set(fov_cells(*B,FOV,G))
TRI=np.array(sorted(a&c&b)); AC=np.array(sorted(a&c))
Bmask=np.zeros((G,G),bool)
for r,cc in b: Bmask[r,cc]=True

def cert_of(nu,al,be):
    epi=be/(np.maximum(nu,1e-6)*np.maximum(al-1.0,1e-6))
    return 1.0/(1.0+np.minimum(epi,10.0)), epi

def one(d,seed):
    d=Path(d)
    R=load_run(d,seed); T=R["T"]
    m=metrics(R,last=LAST,mask=Bmask)["averaged"]
    bel=np.load(d/"cell_beliefs_steps.npy").astype(np.float64)
    nc=np.load(d/"cell_ncov_steps.npy").astype(int)
    gam,nu_s,al_s,be_s=averaged_params(bel,nc)
    cert,_=cert_of(nu_s,al_s,be_s)
    e2=R["averaged"][0]
    cm=np.where(Bmask[None],e2,np.nan); cc_=np.where(Bmask[None],cert,np.nan)
    r_mean,_,_,_=per_timestep_then_averaged(cm,cc_,T-LAST,T)
    # ---- evidence asymmetry on the shared cells ----
    def spread(cells):
        nu=bel[:,cells[:,0],cells[:,1],:,1]; g=bel[:,cells[:,0],cells[:,1],:,0]
        fin=np.isfinite(g)
        n=fin.sum(-1)
        ok=n>=2
        nucv=np.where(ok,np.nanstd(np.where(fin,nu,np.nan),-1)/
                      np.maximum(np.nanmean(np.where(fin,nu,np.nan),-1),1e-12),np.nan)
        x=np.where(fin,wrap_np(g-g[...,0][...,None]),np.nan)
        gs=np.where(ok,np.nanmax(x,-1)-np.nanmin(x,-1),np.nan)*180
        return float(np.nanmean(nucv[-LAST:])), float(np.nanmean(gs[-LAST:]))
    nucv_tri,gs_tri=spread(TRI); nucv_ac,gs_ac=spread(AC)
    # ---- anchor divergence: |epi_A - epi_C| on A&C, slots 0 and 1 ----
    nuA,alA,beA=(bel[:,AC[:,0],AC[:,1],0,i] for i in (1,2,3))
    nuC,alC,beC=(bel[:,AC[:,0],AC[:,1],1,i] for i in (1,2,3))
    _,epiA=cert_of(nuA,alA,beA); _,epiC=cert_of(nuC,alC,beC)
    ok=np.isfinite(epiA)&np.isfinite(epiC)
    div=float(np.nanmean(np.abs(epiA-epiC)[-LAST:][ok[-LAST:]]))
    rel=float(np.nanmean((np.abs(epiA-epiC)/np.maximum((epiA+epiC)/2,1e-12))[-LAST:][ok[-LAST:]]))
    return dict(**m, r=r_mean, nucv_tri=nucv_tri, gs_tri=gs_tri,
                nucv_ac=nucv_ac, gs_ac=gs_ac, div=div, rel=rel,
                epiA=float(np.nanmean(epiA[-LAST:])), epiC=float(np.nanmean(epiC[-LAST:])))

D={}
for arm in ARMS:
    V=[]
    for p in sorted(glob.glob("%s/%s_seed*/manifest.yaml"%(ROOT,arm))):
        d=Path(os.path.dirname(p)); s=int(yaml.safe_load(open(p))["env_seed"])
        V.append(one(d,s))
    if V: D[arm]=V
ms=lambda V,k:(st.mean([v[k] for v in V]), st.stdev([v[k] for v in V]) if len(V)>1 else 0.0)

print("="*104)
print("SECTION 5.3 -- THREE-NODE COLOUR WORLD.  Metrics over node B's full 49-cell FOV")
print("averaged readout, cell space, 5 seeds, mean +/- sd")
print("="*104)
print("%-24s %3s %-20s %-20s %-15s %-15s %-15s"
      % ("arm","n","whole-run MSE","last-50 MSE","90% coverage","hw : RMS","cert-MSE r"))
for arm in ARMS:
    if arm not in D: continue
    V=D[arm]; f=lambda k,p=5:"%.*f +/- %.*f"%(p,ms(V,k)[0],p,ms(V,k)[1])
    print("%-24s %3d %-20s %-20s %-15s %-15s %-15s"
          % (arm,len(V),f("whole"),f("last50"),f("cov",3),f("ratio",3),f("r",3)))

print("\n"+"="*104)
print("EVIDENCE ASYMMETRY -- did the colour world create it?")
print("="*104)
print("%-24s %-22s %-22s %-22s"
      % ("arm","nu-CV (triple, 12)","gamma spread deg","nu-CV (A&C, 21)"))
for arm in ARMS:
    if arm not in D: continue
    V=D[arm]; f=lambda k,p=3:"%.*f +/- %.*f"%(p,ms(V,k)[0],p,ms(V,k)[1])
    print("%-24s %-22s %-22s %-22s" % (arm,f("nucv_tri"),f("gs_tri",2),f("nucv_ac")))
print("\n  offset-world homogeneous reference: nu-CV 0.101, gamma spread 2-5 deg")
print("\nANCHOR DIVERGENCE on the A&C shared cells (last-50)")
print("%-24s %14s %14s %16s %14s" % ("arm","mean epi A","mean epi C","|epiA-epiC|","relative"))
for arm in ARMS:
    if arm not in D: continue
    V=D[arm]
    print("%-24s %14.5f %14.5f %16.5f %13.1f%%"
          % (arm,ms(V,"epiA")[0],ms(V,"epiC")[0],ms(V,"div")[0],100*ms(V,"rel")[0]))
