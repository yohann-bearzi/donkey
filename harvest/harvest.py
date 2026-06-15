#!/usr/bin/env python3
"""Donkey trace harvest from MiMo-V2.5 (block_fp8) on MLX. Generated decode only."""
import argparse, gc, glob, json, os, time
import numpy as np

BASE   = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
MODEL  = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
VERIFY = os.path.expanduser("~/projects/mlx-block-fp8/mimo_prompt.npz")
VERIFY_TOK = 151667
CORPORA = {"humaneval":"humaneval","mbpp":"mbpp","codealpaca":"codealpaca_20k","codealpaca_20k":"codealpaca_20k"}
STOP = (151645, 151643)
CAP_DEFAULT = 32768
TEMP, TOP_P = 1.0, 0.95
ENERGY = 0.99; NUC_CAP = 512; SEED = 0; NEG = -1e30

def load_offsets(d):
    p = np.fromfile(os.path.join(d,"prompts.bin"), dtype=np.int32)
    o = np.fromfile(os.path.join(d,"offsets.bin"), dtype=np.int32)
    assert o.size>1 and o[0]==0 and o[-1]==p.size, f"bad offsets in {d}"
    return p, o

def load_model(mx):
    import mlx_lm.utils as U
    mx.set_wired_limit(300*1024**3)
    cfg = json.load(open(os.path.join(MODEL,"config.json"))); cfg["model_type"]="mimo_v2_block_fp8"
    mc, ac = U._get_classes(cfg); m = mc(ac.from_dict(cfg))
    shards = sorted(glob.glob(os.path.join(MODEL,"*.safetensors")))
    assert shards, f"no .safetensors in {MODEL}"
    w = {}
    for s in shards: w.update(mx.load(s))
    w = m.sanitize_block_fp8(w); m.apply_block_fp8(w); del w
    gc.collect(); mx.eval(m.parameters()); return m, cfg

def gate(mx, m):
    ids = np.load(VERIFY)["ids"].astype(np.int32)
    c = m.make_cache(); lg = m(mx.array(ids[None,:]), cache=c); mx.eval(lg)
    f = int(mx.argmax(lg[0,-1,:]).item())
    if f != VERIFY_TOK: raise SystemExit(f"GATE FAIL: {f} != {VERIFY_TOK}")
    print(f"[gate] ok -> {f}", flush=True)

def install_capture(m):
    inner = m.model; norm = inner.norm
    assert callable(norm), "m.model.norm missing"
    stash={}; orig=norm
    def cap(x,*a,**k): stash["h"]=x; return orig(x,*a,**k)
    inner.norm = cap; return lambda: stash["h"]

class Writer:
    def __init__(s, out, H):
        s.out,s.H=out,H; os.makedirs(out,exist_ok=True)
        s.names=["tokens.bin","lastHiddenState.bin","prompt_idx.bin","topp_counts.bin","topp_ids.bin","topp_probs.bin"]
        s.pp=os.path.join(out,"progress.json"); s.pos=0;s.ent=0;s.done=0;s.cap=0; s._resume()
        s.fh={n:open(os.path.join(out,n),"ab") for n in s.names}
    def _bytes(s):
        return {"tokens.bin":s.pos*4,"lastHiddenState.bin":s.pos*s.H*4,"prompt_idx.bin":s.pos*4,
                "topp_counts.bin":s.pos*4,"topp_ids.bin":s.ent*4,"topp_probs.bin":s.ent*2}
    def _resume(s):
        if not os.path.exists(s.pp):
            for n in s.names: open(os.path.join(s.out,n),"ab").close()
            s._save(); return
        pr=json.load(open(s.pp)); s.done=pr["n_prompts_complete"]; s.pos=pr["total_positions"]
        s.ent=pr["total_topp_entries"]; s.cap=pr["n_prompts_cap_hit"]
        for n,b in s._bytes().items():
            p=os.path.join(s.out,n); open(p,"ab").close()
            with open(p,"r+b") as f: f.truncate(b)
        print(f"[resume] {s.done} prompts, {s.pos} positions", flush=True)
    def _save(s):
        pr={"n_prompts_complete":s.done,"total_positions":s.pos,"total_topp_entries":s.ent,"n_prompts_cap_hit":s.cap}
        t=s.pp+".tmp"; json.dump(pr,open(t,"w")); os.replace(t,s.pp)
    def write(s,toks,hid,pidx,cnt,ids,probs,ch):
        n=toks.shape[0]
        toks.astype(np.int32).tofile(s.fh["tokens.bin"])
        hid.astype(np.float32).tofile(s.fh["lastHiddenState.bin"])
        np.full(n,pidx,np.int32).tofile(s.fh["prompt_idx.bin"])
        cnt.astype(np.int32).tofile(s.fh["topp_counts.bin"])
        ids.astype(np.int32).tofile(s.fh["topp_ids.bin"])
        probs.astype(np.float16).tofile(s.fh["topp_probs.bin"])
        for f in s.fh.values(): f.flush(); os.fsync(f.fileno())
        s.pos+=n; s.ent+=int(ids.shape[0]); s.done+=1; s.cap+=int(ch); s._save()
    def close(s):
        for f in s.fh.values(): f.close()

def run_prompt(mx, m, ids, get_h, temp, cap, H, stream_tok=None, loopguard=8):
    cache=m.make_cache(); toks=[];hids=[];cnt=[];nids=[];nprob=[]
    def step(x):
        lg=m(x,cache=cache); h=get_h()
        L=lg[0,-1,:].astype(mx.float32); Hv=h[0,-1,:].astype(mx.float32)
        idx_desc=mx.argsort(-L); itop=idx_desc[:NUC_CAP]; ptop=mx.softmax(L)[itop]
        if temp<=0:
            tok=idx_desc[0]
        else:
            order=mx.argsort(L); p=mx.softmax(L[order])
            cdf=mx.cumsum(p); keep=cdf>(1.0-TOP_P)
            sc=mx.where(keep,L[order],NEG)+mx.random.gumbel((L.shape[0],)); tok=order[mx.argmax(sc)]
        mx.eval(tok,Hv,ptop,itop)
        ip=np.array(itop); pp=np.array(ptop)
        c=min(int(np.searchsorted(np.cumsum(pp),ENERGY)+1), pp.shape[0])
        return int(tok.item()), np.array(Hv,copy=True), ip[:c], pp[:c]
    tok,_,_,_=step(mx.array(ids[None,:].astype(np.int32)))   # prefill: not recorded
    n=1; loop=False
    while tok not in STOP and n<cap:
        tok,Hv,ii,pp=step(mx.array([[tok]],dtype=mx.int32))
        toks.append(tok);hids.append(Hv);cnt.append(ii.shape[0]);nids.append(ii);nprob.append(pp);n+=1
        if stream_tok is not None and tok not in STOP:
            print(stream_tok.decode([tok]),end="",flush=True)
        if loopguard>0 and len(toks)>=100 and len(set(toks[-100:]))<loopguard:
            loop=True; break
    if not toks:
        return (np.zeros(0,np.int32),np.zeros((0,H),np.float32),np.zeros(0,np.int32),
                np.zeros(0,np.int32),np.zeros(0,np.float32),True)
    ch = (toks[-1] not in STOP) or loop
    if stream_tok is not None: print(flush=True)
    return (np.array(toks,np.int32),np.stack(hids).astype(np.float32),np.array(cnt,np.int32),
            np.concatenate(nids),np.concatenate(nprob),ch)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("corpus",choices=sorted(CORPORA))
    ap.add_argument("--smoke",type=int,default=None); ap.add_argument("--cap",type=int,default=CAP_DEFAULT)
    ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--stream",action="store_true",help="print tokens live")
    ap.add_argument("--loopguard",type=int,default=8,help="min unique tokens in last 100; <1 disables")
    a=ap.parse_args()
    import mlx.core as mx; mx.random.seed(SEED)
    name=CORPORA[a.corpus]; pdir=os.path.join(BASE,"prompts",name); odir=os.path.join(BASE,"traces",name)
    logdir=os.path.join(odir,"decoded"); os.makedirs(logdir,exist_ok=True); logf=os.path.join(logdir,f"{name}.log")
    prompts,off=load_offsets(pdir); nP=off.size-1; print(f"[corpus] {name}: {nP} prompts",flush=True)
    temp=0.0 if a.smoke is not None else TEMP
    m,cfg=load_model(mx); H=int(cfg.get("hidden_size",4096)); gate(mx,m)
    try:
        from tokenizers import Tokenizer as _TK; tok_=_TK.from_file(os.path.join(MODEL,'tokenizer.json')); tok_.decode=tok_.decode; have_tok=True
    except Exception as e:
        print(f"[warn] tokenizer load failed, logging ids only: {e}",flush=True); have_tok=False
    get_h=install_capture(m); W=Writer(odir,H)
    end=min(a.smoke,nP) if a.smoke is not None else nP
    if a.limit is not None: end=min(end,a.limit)
    t0=time.time(); ntok=0
    for i in range(W.done,end):
        pids=prompts[off[i]:off[i+1]]; s=time.time()
        stream_tok = tok_ if (have_tok and (a.stream or a.smoke is not None)) else None
        print(f'\n----- prompt {i} -----',flush=True) if stream_tok is not None else None
        toks,hid,cnt,ii,pp,ch=run_prompt(mx,m,pids,get_h,temp,a.cap,H,stream_tok,a.loopguard)
        W.write(toks,hid,i,cnt,ii,pp,ch); dt=time.time()-s; ntok+=toks.shape[0]
        body = tok_.decode([int(t) for t in toks if int(t) not in STOP]) if (have_tok and toks.size) else ""
        with open(logf,"a") as lf:
            lf.write(f"\n===== prompt {i} | {toks.shape[0]} tok | {dt:.1f}s | {toks.shape[0]/max(dt,1e-9):.1f} tok/s | {'CAP' if ch else 'STOP'} =====\n{body}\n")
        print(f"[{i+1}/{end}] {toks.shape[0]} tok {dt:.1f}s ({toks.shape[0]/max(dt,1e-9):.1f} tok/s){' CAP' if ch else ''}",flush=True)
    W.close()
    meta={"convention":"self-aligned: hidden[i]->tokens[i]; generated decode only (prefill excluded)",
          "hidden_layer":"pre-final-rmsnorm","hidden_size":H,"trunk_model":"MiMo-V2.5-block_fp8",
          "sampler":{"temperature":temp,"top_p":TOP_P},"nucleus_energy":ENERGY,"nucleus_cap":NUC_CAP,
          "nucleus_probs":"raw temp=1.0, not renormalized","seed":SEED,"max_gen_cap":a.cap,
          "stop_tokens":list(STOP),"total_positions":W.pos,"total_topp_entries":W.ent,
          "n_prompts_complete":W.done,"n_prompts_cap_hit":W.cap}
    json.dump(meta,open(os.path.join(odir,"meta.json"),"w"),indent=2)
    print(f"[done] {W.done} prompts, {W.pos} positions, {ntok/max(time.time()-t0,1e-9):.1f} tok/s",flush=True)
    if a.smoke is not None:
        t=np.fromfile(os.path.join(odir,"tokens.bin"),dtype=np.int32)
        h=np.fromfile(os.path.join(odir,"lastHiddenState.bin"),dtype=np.float32).reshape(-1,H)
        if h.shape[0]:
            lm=getattr(m,"lm_head",None) or m.model.embed_tokens
            k=min(48,h.shape[0]); idx=np.random.default_rng(0).choice(h.shape[0],k,replace=False)
            pred=np.array(mx.argmax(lm(m.model.norm(mx.array(h[idx]))),axis=-1)); ag=float((pred==t[idx]).mean())
            print(f"[smoke] self-alignment (greedy): {ag*100:.1f}% argmax==token -> {'PASS' if ag==1.0 else 'FAIL'}",flush=True)
        print(f"[smoke] decoded text -> {logf}",flush=True)

if __name__=="__main__": main()
