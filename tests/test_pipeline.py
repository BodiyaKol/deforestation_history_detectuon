"""tests/test_pipeline.py — повний тестовий пайплайн"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main_logic_SVD.svd_decomposition import compute_svd_background
from main_logic_SVD.forest_masks      import build_forest_masks
from main_logic_SVD.anomaly_detection import compute_anomalies
from main_logic_SVD.spatial_filter    import filter_spatial_noise
from main_logic_SVD.diagnostics       import plot_diagnostics

# ── helpers ──────────────────────────────────────────────────────────────────
def _scene(H=60,W=60,T=20,noise=0.01,seed=0):
    rng=np.random.default_rng(seed); m=max(2,H//6)
    base=np.full((H,W),0.12); base[m:H-m,m:W-m]=0.65
    s=0.04*np.sin(2*np.pi*np.arange(T)/T)
    X=np.stack([base+s[t]+rng.normal(0,noise,(H,W)) for t in range(T)]).reshape(T,H*W).T
    return np.clip(X,-1,1), base

def _inject(X,H,W,r0,r1,c0,c1,t0,T,ndvi=0.03,seed=42):
    rng=np.random.default_rng(seed); Xm=X.copy()
    gt=np.zeros((H,W),dtype=bool); gt[r0:r1,c0:c1]=True; flat=gt.flatten()
    for t in range(t0,T): Xm[flat,t]=ndvi+rng.normal(0,0.005,flat.sum())
    return Xm, gt

# ═══ UNIT TESTS ══════════════════════════════════════════════════════════════
def t1_clean():
    X,_=_scene(noise=0.005)
    L,S,sig,k=compute_svd_background(X,variance_threshold=0.999)
    v=np.abs(S).max()
    assert v<0.05, f"max|S|={v:.4f}>0.05 k={k}"
    print(f"  [PASS] svd_clean_scene          max|S|={v:.5f}  k={k}")

def t2_numpy_match():
    X,_=_scene(H=30,W=30,T=10)
    L_ours,_,_,k=compute_svd_background(X,variance_threshold=0.99)
    U,s,Vt=np.linalg.svd(X,full_matrices=False)
    s2c=np.cumsum(s**2)/np.sum(s**2); kr=int(np.searchsorted(s2c,0.99))+1
    L_ref=U[:,:kr]@np.diag(s[:kr])@Vt[:kr,:]
    err=np.linalg.norm(L_ours-L_ref)/(np.linalg.norm(L_ref)+1e-8)
    assert err<0.05, f"rel_err={err:.4f}>5% k={k} kr={kr}"
    print(f"  [PASS] svd_matches_numpy        rel_err={err:.4f}  k={k}")

def t3_seasonal():
    H,W,T=40,40,20; rng=np.random.default_rng(1)
    base=np.full((H,W),0.65); s=0.15*np.sin(2*np.pi*np.arange(T)/T)
    X=np.stack([base+s[t]+rng.normal(0,0.008,(H,W)) for t in range(T)]).reshape(T,H*W).T
    L,S,sig,k=compute_svd_background(X,variance_threshold=0.999)
    fm=np.ones((H,W),dtype=bool); nfm=np.zeros((H,W),dtype=bool)
    _,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-2.5)
    fpr=am.mean()
    assert fpr<0.02, f"FPR={fpr:.3%}>2% k={k}"
    print(f"  [PASS] seasonal_not_anomaly     FPR={fpr:.3%}  k={k}")

def t4_recall():
    H,W,T=60,60,20; X,_=_scene(H=H,W=W,T=T,noise=0.008)
    Xm,gt=_inject(X,H,W,22,32,22,32,T//2,T,ndvi=0.03)
    L,S,sig,k=compute_svd_background(Xm)
    fm,nfm=build_forest_masks(Xm,H,W); _,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-1.7)
    det=am[:,T//2:].any(axis=1).reshape(H,W); gtf=gt&fm
    rec=(det&gtf).sum()/(gtf.sum()+1e-8)
    assert rec>=0.70, f"recall={rec:.2%}<70% k={k}"
    print(f"  [PASS] deforestation_recall     recall={rec:.2%}  k={k}")

def t5_spatial():
    H,W,T=50,50,8; rng=np.random.default_rng(7)
    nm=rng.random((H*W,T))<0.02; sig2d=np.zeros((H,W),dtype=bool); sig2d[20:28,20:28]=True
    flat=sig2d.flatten()
    for t in range(3,T): nm[flat,t]=True
    filt,_=filter_spatial_noise(nm,H,W,min_cluster_size=10)
    sk=filt[flat,3:].mean(); nm2=nm.copy(); nm2[flat,:]=False
    nk=filt[nm2].mean() if nm2.any() else 0.0
    assert sk>=0.80, f"signal={sk:.2%}<80%"
    assert nk<0.10,  f"noise={nk:.2%}>10%"
    print(f"  [PASS] spatial_filter           signal={sk:.0%}  noise={nk:.0%}")

def t6_precision():
    H,W,T=60,60,20; X,_=_scene(H=H,W=W,T=T,noise=0.008)
    Xm,gt=_inject(X,H,W,38,48,38,48,T//2,T,ndvi=0.03)
    L,S,sig,k=compute_svd_background(Xm)
    fm,nfm=build_forest_masks(Xm,H,W); _,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-2.0)
    acl,_=filter_spatial_noise(am,H,W,min_cluster_size=20)
    det=acl[:,T//2:].any(axis=1).reshape(H,W)
    TP=(det&gt).sum(); FP=(det&~gt).sum(); prec=TP/(TP+FP+1e-8)
    assert prec>=0.80, f"precision={prec:.2%}<80% TP={TP} FP={FP}"
    print(f"  [PASS] precision_spatial        precision={prec:.2%}  TP={TP}  FP={FP}")

def t7_nonforest():
    H,W,T=40,40,10; X,_=_scene(H=H,W=W,T=T); Xm=X.copy()
    ff=np.zeros((H,W),dtype=bool); ff[0:6,0:6]=True
    for t in range(5,T): Xm[ff.flatten(),t]=0.0
    L,S,sig,k=compute_svd_background(Xm); fm,nfm=build_forest_masks(Xm,H,W)
    _,Z,am=compute_anomalies(S,fm,nfm,H,W); n=am[ff.flatten(),:].sum()
    assert n==0, f"{n} аномалій на полях"
    print(f"  [PASS] nonforest_ignored        аномалій на полях={n}")

def t8_gradual():
    H,W,T=50,50,20; X,_=_scene(H=H,W=W,T=T)
    gt=np.zeros((H,W),dtype=bool); gt[15:25,15:25]=True
    flat=gt.flatten(); rng=np.random.default_rng(9); Xm=X.copy()
    for t in range(T//2,T):
        p=(t-T//2)/(T-T//2); Xm[flat,t]=(0.65-p*0.50)+rng.normal(0,0.008,flat.sum())
    L,S,sig,k=compute_svd_background(Xm); fm,nfm=build_forest_masks(Xm,H,W)
    _,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-1.8)
    dl=am[:,-4:].any(axis=1).reshape(H,W); rec=(dl&gt&fm).sum()/((gt&fm).sum()+1e-8)
    assert rec>=0.50, f"gradual recall={rec:.2%}<50%"
    print(f"  [PASS] gradual_degradation      recall={rec:.2%}  k={k}")

def t9_scattered():
    H,W,T=60,60,20; X,_=_scene(H=H,W=W,T=T)
    spots=[(12,12),(12,35),(35,12)]; sz=4
    gt=np.zeros((H,W),dtype=bool); rng=np.random.default_rng(11); Xm=X.copy()
    for r0,c0 in spots: gt[r0:r0+sz,c0:c0+sz]=True
    for t in range(T//2,T): Xm[gt.flatten(),t]=0.03+rng.normal(0,0.005,gt.sum())
    L,S,sig,k=compute_svd_background(Xm); fm,nfm=build_forest_masks(Xm,H,W)
    _,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-1.8)
    det=am[:,T//2:].any(axis=1).reshape(H,W)
    found=sum((det[r0:r0+sz,c0:c0+sz]&gt[r0:r0+sz,c0:c0+sz]).any() for r0,c0 in spots)
    assert found>=1, f"0 плям знайдено з {len(spots)}"
    print(f"  [PASS] small_scattered          знайдено {found}/{len(spots)} плям")

UNIT_TESTS=[t1_clean,t2_numpy_match,t3_seasonal,t4_recall,t5_spatial,t6_precision,t7_nonforest,t8_gradual,t9_scattered]

# ═══ INTEGRATION TESTS ═══════════════════════════════════════════════════════
def _metrics(detected,gt_3d,event_start):
    gt_any=gt_3d[:,:,event_start:].any(axis=2)
    TP=(detected&gt_any).sum(); FP=(detected&~gt_any).sum(); FN=(~detected&gt_any).sum()
    p=TP/(TP+FP+1e-8); r=TP/(TP+FN+1e-8); f1=2*p*r/(p+r+1e-8)
    return dict(TP=int(TP),FP=int(FP),FN=int(FN),precision=float(p),recall=float(r),f1=float(f1))

def _plot_detection(X,L,Z,am_clean,gt_3d,H,W,T,dates,sigma,k,out_dir,name):
    plot_diagnostics(sigma,X,L,Z,am_clean,H,W,dates,k,out_dir)
    gt_any  = gt_3d.any(axis=2)
    det_any = am_clean.any(axis=1).reshape(H,W)

    fig,axes=plt.subplots(1,4,figsize=(20,5))
    fig.suptitle(f"Detection — {name}",fontsize=13,fontweight="bold")

    axes[0].imshow(X[:,0].reshape(H,W),cmap="YlGn",vmin=0,vmax=0.8)
    axes[0].set_title("Кадр 0 — початок"); axes[0].axis("off")

    axes[1].imshow(X[:,-1].reshape(H,W),cmap="YlGn",vmin=0,vmax=0.8)
    axes[1].set_title(f"Кадр {T-1} — кінець"); axes[1].axis("off")

    axes[2].imshow(gt_any.astype(float),cmap="Reds",vmin=0,vmax=1)
    axes[2].set_title("Ground Truth"); axes[2].axis("off")

    overlay=np.zeros((H,W,3))
    overlay[:,:,1]=gt_any.astype(float)*0.7   # зелений=gt
    overlay[:,:,0]=det_any.astype(float)*0.9  # червоний=detected
    axes[3].imshow(overlay)
    axes[3].set_title("🟢 GT   🔴 Detected   🟡 TP")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir/"detection.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"    → detection.png saved")

def run_one(image_dir,results_base):
    X    =np.load(image_dir/"X.npy")
    meta =np.load(image_dir/"meta.npy",allow_pickle=True).item()
    gt_3d=np.load(image_dir/"ground_truth.npy")
    info =(image_dir/"scenario_info.txt").read_text()
    H,W=meta["height"],meta["width"]; dates=meta["dates"]; T=len(dates); name=image_dir.name

    print(f"\n  {'─'*50}")
    print(f"  {name}  [{H}×{W}, T={T}]")

    L,S,sigma,k=compute_svd_background(X)
    fm,nfm=build_forest_masks(X,H,W)
    Sm,Z,am=compute_anomalies(S,fm,nfm,H,W,z_threshold=-2.0)
    am_cl,_=filter_spatial_noise(am,H,W,min_cluster_size=8)
    detected=am_cl.any(axis=1).reshape(H,W)
    m=_metrics(detected,gt_3d,T//2)

    out_dir=results_base/name; out_dir.mkdir(parents=True,exist_ok=True)
    _plot_detection(X,L,Z,am_cl,gt_3d,H,W,T,dates,sigma,k,out_dir,name)

    lines=[f"Scenario: {name}",f"Size:{H}x{W} T:{T} k:{k}","",
           "--- Metrics ---",f"  Precision : {m['precision']:.3f}",
           f"  Recall    : {m['recall']:.3f}",f"  F1        : {m['f1']:.3f}",
           f"  TP={m['TP']} FP={m['FP']} FN={m['FN']}","","--- Info ---",info]
    (out_dir/"metrics.txt").write_text("\n".join(lines))

    mark="✓" if m["f1"]>0.30 else "△"
    print(f"  {mark} P={m['precision']:.2%}  R={m['recall']:.2%}  F1={m['f1']:.2%}")
    return {"name":name,**m}

def run_integration(images_dir,results_dir):
    dirs=sorted(d for d in Path(images_dir).glob("*/") if (d/"X.npy").exists())
    if not dirs:
        print(f"\n[!] Немає папок у {images_dir}")
        print("    Запусти: python tests/generate_test_images.py --all"); return False
    print("\n"+"═"*55); print(f"  INTEGRATION TESTS  ({len(dirs)} сценаріїв)"); print("═"*55)
    all_m=[]
    for d in dirs:
        all_m.append(run_one(d,Path(results_dir)))
    print("\n"+"─"*55)
    print(f"  {'Сценарій':<32} {'P':>6} {'R':>6} {'F1':>6}")
    print("─"*55)
    for m in all_m:
        mk="✓" if m["f1"]>0.30 else "△"
        print(f"  {mk} {m['name']:<30} {m['precision']:>5.0%} {m['recall']:>5.0%} {m['f1']:>5.0%}")
    print("─"*55)
    summary=["SUMMARY","="*50]+[f"{m['name']:<32} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}" for m in all_m]
    Path(results_dir).mkdir(parents=True,exist_ok=True)
    (Path(results_dir)/"summary.txt").write_text("\n".join(summary))
    print(f"\n  → {results_dir}/summary.txt")
    return True

# ═══ MAIN ════════════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--unit",action="store_true")
    ap.add_argument("--integration",action="store_true")
    ap.add_argument("--images-dir",default="tests/test_images")
    ap.add_argument("--results-dir",default="tests/test_results")
    args=ap.parse_args()
    run_u=args.unit or not(args.unit or args.integration)
    run_i=args.integration or not(args.unit or args.integration)
    ok=True
    if run_u:
        print("\n"+"═"*55); print("  UNIT TESTS"); print("═"*55)
        p=f=0
        for fn in UNIT_TESTS:
            try: fn(); p+=1
            except AssertionError as e: print(f"  [FAIL] {fn.__name__}: {e}"); f+=1
            except Exception as e: print(f"  [ERR ] {fn.__name__}: {e}"); f+=1
        print(f"\n  Результат: {p}/{p+f} passed")
        ok=ok and f==0
    if run_i:
        ok=run_integration(args.images_dir,args.results_dir) and ok
    sys.exit(0 if ok else 1)

if __name__=="__main__":
    main()