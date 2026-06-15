#!/usr/bin/env python3
"""donkey_train v3 - JOINT latent world model over MiMo-V2.5 hiddens (MLX)."""
import argparse, json, os, time, glob
import numpy as np

BASE   = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
MODEL  = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
CKPTS  = os.environ.get("DONKEY_CKPTS", os.path.expanduser("~/projects/donkey/ckpts"))
CORPORA = {"humaneval":"humaneval","mbpp":"mbpp","codealpaca":"codealpaca_20k","codealpaca_20k":"codealpaca_20k"}
STOP = (151645, 151643)
DMODEL = 4096; SEED = 0
SIGREG_M = 1024
WINDOW = 13; PRED_LAYERS = 6; ENC_LAYERS = 2; DEC_LAYERS = 2
_NORM = {}

def load_trace(corpus):
    d = os.path.join(BASE, "traces", CORPORA[corpus])
    h = np.fromfile(os.path.join(d,"lastHiddenState.bin"), np.float32).reshape(-1, DMODEL)
    tok = np.fromfile(os.path.join(d,"tokens.bin"), np.int32)
    pid = np.fromfile(os.path.join(d,"prompt_idx.bin"), np.int32)
    n = min(h.shape[0], tok.shape[0], pid.shape[0]); h,tok,pid = h[:n],tok[:n],pid[:n]
    mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
    _NORM[corpus] = (mu, sd); h = ((h - mu)/sd).astype(np.float32)
    print(f"[data:{corpus}] {n} pos, std was {float(sd.mean()):.1f} -> unit")
    return h, tok, pid

def load_ipr_weights(corpus):
    d = os.path.join(BASE,"traces",CORPORA[corpus])
    counts = np.fromfile(os.path.join(d,"topp_counts.bin"), np.int32)
    probs = np.fromfile(os.path.join(d,"topp_probs.bin"), np.float16).astype(np.float64)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    w = np.ones(len(counts), np.float64)
    for i in range(len(counts)):
        p = probs[off[i]:off[i+1]]; s = p.sum()
        if s > 0: p = p/s; w[i] = float((p*p).sum())
    return w.astype(np.float32)

def build_transitions(h, tok, pid, W, keep_only_terminated=True):
    N=h.shape[0]; bnd=np.where(np.diff(pid)!=0)[0]+1; spans=np.split(np.arange(N),bnd)
    win,act,tgt,src=[],[],[],[]; nskip=0
    for span in spans:
        if span.size<2: continue
        if keep_only_terminated and int(tok[span[-1]]) not in STOP: nskip+=1; continue
        s0=span[0]
        for j in range(span.size-1):
            idxs=[span[max(0,j-(W-1)+k)] if (j-(W-1)+k)>=0 else s0 for k in range(W)]
            win.append(idxs); act.append(tok[span[j]]); tgt.append(span[j+1]); src.append(span[j])
    if keep_only_terminated and nskip: print(f"[data] skipped {nskip} cap-hit prompts")
    return (np.asarray(win,np.int64),np.asarray(act,np.int32),np.asarray(tgt,np.int64),np.asarray(src,np.int64))

def make_sigreg(mx, M=SIGREG_M):
    knots=mx.linspace(-5.0,5.0,33); tgt=mx.exp(-(knots**2)/2)[None,:]; wt=tgt
    def sigreg(Z):
        Zf=Z.reshape(-1,Z.shape[-1]); d=Zf.shape[-1]
        u=mx.random.normal((d,M)); u=u/mx.sqrt(mx.sum(u*u,axis=0,keepdims=True)+1e-9)
        H=Zf@u; arg=H[:,:,None]*knots[None,None,:]
        re=mx.mean(mx.cos(arg),axis=0); im=mx.mean(mx.sin(arg),axis=0)
        return mx.mean(mx.sum(((re-tgt)**2+im**2)*wt,axis=-1))
    return sigreg

def make_modules(mx, nn, d, dp, n_enc, n_dec, n_actions):
    rootD = float(np.sqrt(d))
    def normD(z):
        return z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-6) * rootD
    class Block(nn.Module):
        def __init__(self,w,adaln=False,heads=8):
            super().__init__()
            self.n1=nn.RMSNorm(w);self.n2=nn.RMSNorm(w)
            self.q=nn.Linear(w,w,bias=False);self.k=nn.Linear(w,w,bias=False)
            self.v=nn.Linear(w,w,bias=False);self.o=nn.Linear(w,w,bias=False)
            self.f1=nn.Linear(w,4*w);self.f2=nn.Linear(4*w,w);self.heads=heads;self.w=w;self.adaln=adaln
            if adaln:
                self.ada=nn.Linear(w,4*w);self.ada.weight=mx.zeros_like(self.ada.weight);self.ada.bias=mx.zeros_like(self.ada.bias)
        def __call__(self,x,a_emb=None,mask=None):
            B,T,W=x.shape
            if self.adaln and a_emb is not None: s1,b1,s2,b2=mx.split(self.ada(a_emb)[:,None,:],4,axis=-1)
            else: s1=b1=s2=b2=0.0
            y=self.n1(x)*(1+s1)+b1; hd=self.w//self.heads
            q=self.q(y).reshape(B,T,self.heads,hd).transpose(0,2,1,3)
            k=self.k(y).reshape(B,T,self.heads,hd).transpose(0,2,1,3)
            v=self.v(y).reshape(B,T,self.heads,hd).transpose(0,2,1,3)
            att=(q@k.transpose(0,1,3,2))*(hd**-0.5)
            if mask is not None: att=att+mask
            att=mx.softmax(att,axis=-1); o=(att@v).transpose(0,2,1,3).reshape(B,T,W); x=x+self.o(o)
            y=self.n2(x)*(1+s2)+b2; return x+self.f2(nn.gelu(self.f1(y)))
    class Encoder(nn.Module):
        def __init__(s):
            super().__init__(); s.inp=nn.Linear(DMODEL,dp); s.blocks=[Block(dp) for _ in range(n_enc)]; s.head=nn.Linear(dp,d)
        def __call__(s,h):
            x=s.inp(h)
            for b in s.blocks: x=b(x)
            return normD(s.head(x))
    class Predictor(nn.Module):
        def __init__(s):
            super().__init__(); s.inp=nn.Linear(d,dp); s.pos=mx.random.normal((WINDOW,dp))*0.02
            s.blocks=[Block(dp,adaln=True) for _ in range(PRED_LAYERS)]; s.head=nn.Linear(dp,d)
        def __call__(s,z,a_emb,mask):
            x=s.inp(z)+s.pos[None,:z.shape[1],:]
            for b in s.blocks: x=b(x,a_emb=a_emb,mask=mask)
            return normD(s.head(x[:,-1,:]))
    class Decoder(nn.Module):
        def __init__(s):
            super().__init__(); s.inp=nn.Linear(d,dp); s.blocks=[Block(dp) for _ in range(n_dec)]; s.head=nn.Linear(dp,DMODEL)
        def __call__(s,z):
            x=s.inp(z)[:,None,:]
            for b in s.blocks: x=b(x)
            return s.head(x[:,0,:])
    return Encoder(), Predictor(), Decoder(), nn.Embedding(n_actions,dp)

def causal_mask(mx,W): return mx.triu(mx.full((W,W),-1e9),k=1)[None,None]
def nparams(mx,m):
    from mlx.utils import tree_flatten
    return sum(v.size for _,v in tree_flatten(m.parameters()))

def load_lm_head(mx):
    cfg=json.load(open(os.path.join(MODEL,"config.json"))); cfg["model_type"]="mimo_v2_block_fp8"
    norm_w=lm_w=emb_w=None
    for s in sorted(glob.glob(os.path.join(MODEL,"*.safetensors"))):
        w=mx.load(s)
        for k,v in w.items():
            if k.endswith("model.norm.weight") or k=="norm.weight": norm_w=v
            if k.endswith("lm_head.weight"): lm_w=v
            if k.endswith("embed_tokens.weight"): emb_w=v
    if lm_w is None: lm_w=emb_w
    assert lm_w is not None and norm_w is not None
    eps=float(cfg.get("rms_norm_eps",1e-6))
    def make_project(mu,sd):
        mu_m=mx.array(mu); sd_m=mx.array(sd)
        def project(h_std):
            h=h_std*sd_m+mu_m; x=h*mx.rsqrt(mx.mean(h*h,axis=-1,keepdims=True)+eps)*norm_w
            return x@lm_w.T
        return project
    return make_project

def save_ckpt(mx,path,module,meta):
    from mlx.utils import tree_flatten
    mx.savez(path, **dict(tree_flatten(module.parameters()))); json.dump(meta,open(path+".meta.json","w"))

def clip_grads(mx,g,maxnorm):
    from mlx.utils import tree_flatten, tree_unflatten
    leaves=[v for _,v in tree_flatten(g)]
    total=mx.sqrt(sum(mx.sum(v*v) for v in leaves)+1e-12)
    scale=mx.minimum(1.0, maxnorm/(total+1e-9))
    return tree_unflatten([(k,v*scale) for k,v in tree_flatten(g)])

def train(args, mx, nn, optim):
    mx.random.seed(SEED); rng=np.random.default_rng(SEED)
    h,tok,pid=load_trace(args.corpus); wi,act,tgt,src=build_transitions(h,tok,pid,WINDOW)
    ipr=load_ipr_weights(args.corpus); tw=ipr[src]
    uniq=np.unique(act); remap={int(t):i for i,t in enumerate(uniq)}; aix=np.array([remap[int(t)] for t in act],np.int32)
    phi,pred,psi,aemb=make_modules(mx,nn,args.d,args.dp,args.enc,DEC_LAYERS,len(uniq))
    sigreg=make_sigreg(mx); mask=causal_mask(mx,WINDOW)
    mu,sd=_NORM[args.corpus]; project=load_lm_head(mx)(mu,sd)
    class WM(nn.Module):
        def __init__(s): super().__init__(); s.phi=phi; s.pred=pred; s.psi=psi; s.aemb=aemb
    wm=WM()
    print(f"[v3 joint] {args.corpus} d={args.d} dp={args.dp} | {nparams(mx,wm)/1e6:.1f}M params | lce={args.lce} lrec={args.lrec} lsig={args.lsig}")
    val_corpus=args.val or args.corpus
    if val_corpus!=args.corpus:
        hv,tokv,pidv=load_trace(val_corpus); wiv,actv,tgtv,srcv=build_transitions(hv,tokv,pidv,WINDOW)
        aixv=np.array([remap.get(int(t),0) for t in actv],np.int32); projv=load_lm_head(mx)(*_NORM[val_corpus])
    else:
        upid=np.unique(pid); rng.shuffle(upid); nval=max(1,int(0.1*len(upid))); vp=set(upid[:nval].tolist())
        vm=np.array([p in vp for p in pid[src]])
        hv,wiv,aixv,tgtv,projv=h,wi[vm],aix[vm],tgt[vm],project
        wi,aix,tgt,tw=wi[~vm],aix[~vm],tgt[~vm],tw[~vm]
    if args.max_transitions and wi.shape[0] > args.max_transitions:
        sel = rng.choice(wi.shape[0], args.max_transitions, replace=False)
        wi, aix, tgt, tw = wi[sel], aix[sel], tgt[sel], tw[sel]
        print(f"[data] capped train to {args.max_transitions} transitions (sweep mode)")
    def fcos(a,b):
        an=a/(mx.linalg.norm(a,axis=-1,keepdims=True)+1e-6); bn=b/(mx.linalg.norm(b,axis=-1,keepdims=True)+1e-6)
        return 1-mx.sum(an*bn,axis=-1)
    def loss(hw,a,ht,w):
        z=wm.phi(hw)
        zt=wm.phi(ht[:,None,:])[:,0,:]
        zhat=wm.pred(z,wm.aemb(a),mask)
        Lcos=mx.sum(w*fcos(zhat,zt))/(mx.sum(w)+1e-6)
        tgt_tok=mx.argmax(project(mx.stop_gradient(ht)),axis=-1)
        Lce =mx.mean(nn.losses.cross_entropy(project(wm.psi(zhat)), tgt_tok))
        Lrec=mx.mean(nn.losses.cross_entropy(project(wm.psi(zt)),   tgt_tok))
        Lsig=sigreg(z)
        return Lcos + args.lce*Lce + args.lrec*Lrec + args.lsig*Lsig
    lg=nn.value_and_grad(wm,loss); opt=optim.AdamW(learning_rate=args.lr)
    def accept_on(hh, wiX, aixX, tgtX, proj):
        k=min(4096,wiX.shape[0]); b=rng.choice(wiX.shape[0],k,replace=False)
        hw=mx.array(hh[wiX[b]]); a=mx.array(aixX[b]); ht=mx.array(hh[tgtX[b]])
        zhat=wm.pred(wm.phi(hw),wm.aemb(a),mask); e2e=np.array(mx.argmax(proj(wm.psi(zhat)),axis=-1))
        trunk=np.array(mx.argmax(proj(ht),axis=-1)); return float((e2e==trunk).mean())
    def val_accept():   return accept_on(hv, wiv, aixv, tgtv, projv)
    def train_accept(): return accept_on(h,  wi,  aix,  tgt,  project)
    best=None;bad=0;bs=args.bs;T=wi.shape[0]
    for ep in range(args.epochs):
        t0=time.time(); idx=rng.permutation(T); tot=nb=0
        for i in range(0,T-bs+1,bs):
            b=idx[i:i+bs]; hw=mx.array(h[wi[b]]);a=mx.array(aix[b]);ht=mx.array(h[tgt[b]]);w=mx.array(tw[b])
            L,g=lg(hw,a,ht,w); g=clip_grads(mx,g,1.0); opt.update(wm,g); mx.eval(wm.parameters(),opt.state); tot+=float(L);nb+=1
        acc=val_accept(); tra=train_accept(); imp=best is None or acc>best
        if imp: best=acc;bad=0; save_ckpt(mx,args.out,wm,{"d":args.d,"dp":args.dp,"enc":args.enc,"val":val_corpus,"n_actions":len(uniq)})
        else: bad+=1
        print(f"[v3 ep{ep+1}/{args.epochs}] loss {tot/nb:.4f} | train {tra*100:.2f}% | val({val_corpus}) {acc*100:.2f}%{' *' if imp else ''} | gap {(tra-acc)*100:+.1f} | bad {bad} | {time.time()-t0:.1f}s")
        if bad>=args.patience: print("[v3] early stop"); break
    print(f"[v3] best val accept {best*100:.2f}% -> {args.out}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("corpus",choices=sorted(CORPORA))
    ap.add_argument("--d",type=int,default=64); ap.add_argument("--dp",type=int,default=256)
    ap.add_argument("--enc",type=int,default=ENC_LAYERS); ap.add_argument("--val",type=str,default=None)
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--bs",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--patience",type=int,default=15)
    ap.add_argument("--lce",type=float,default=1.0); ap.add_argument("--lrec",type=float,default=0.5)
    ap.add_argument("--lsig",type=float,default=0.1); ap.add_argument("--out",type=str,default=None)
    ap.add_argument("--max-transitions",type=int,default=0,help="cap train transitions for sweep speed (0=all)")
    a=ap.parse_args(); os.makedirs(CKPTS,exist_ok=True)
    if a.out is None: a.out=os.path.join(CKPTS,f"v3_{a.corpus}_d{a.d}_dp{a.dp}.npz")
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    train(a,mx,nn,optim)

if __name__=="__main__": main()
