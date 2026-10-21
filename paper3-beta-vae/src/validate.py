"""Fast independent math/architecture checks + optional real GPU throughput benchmark."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import DSprites, DEFAULT_DATA, FACTOR_SIZES, N_IMAGES, index_from_factors, factors_from_index
from .models import BetaVAE, ModelConfig, loss_terms
from .evaluate import load_inference


def smoke(paper_model: bool = False, data: Path | None = None, cache_dir: Path | None = None):
    config = ModelConfig(hidden_dim=1200 if paper_model else 24)
    torch.manual_seed(2026)
    model=BetaVAE(config)
    params=sum(p.numel() for p in model.parameters())
    x=torch.randint(0,2,(2,4096),dtype=torch.float32)
    logits,mu,logvar=model(x)
    assert logits.shape==(2,4096) and mu.shape==logvar.shape==(2,10)
    loss,recon,kl=loss_terms(logits,x,mu,logvar,4)
    ref=F.binary_cross_entropy_with_logits(logits,x,reduction='sum')/2
    refkl=(-0.5*(1+logvar-mu.square()-logvar.exp())).sum()/2
    assert torch.allclose(recon,ref,atol=1e-5)
    assert torch.allclose(kl,refkl,atol=1e-5)
    l1,_,_=loss_terms(logits,x,mu,logvar,1)
    assert torch.allclose(loss-l1,3*kl,atol=1e-4)
    assert abs(float(kl.item())) > 1e-7
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    # Gaussian KL should be zero when posterior equals standard unit Gaussian.
    _,_,zero=loss_terms(logits.detach(),x,torch.zeros_like(mu),torch.zeros_like(logvar),4)
    assert float(zero)==0.0
    with tempfile.TemporaryDirectory() as folder:
        filename=Path(folder)/'model.pt'
        torch.save({'model':model.state_dict(),'model_config':config.__dict__,'beta':4.,'seed':2026,'step':0},filename)
        reloaded,_=load_inference(filename,torch.device('cpu'))
        torch.testing.assert_close(model(x,deterministic=True)[0],reloaded(x,deterministic=True)[0])
    rng=np.random.default_rng(999)
    ids=rng.integers(0,N_IMAGES,size=200)
    np.testing.assert_array_equal(index_from_factors(factors_from_index(ids)),ids)
    assert tuple(FACTOR_SIZES)==(1,3,6,40,32,32)
    print(json.dumps({'status':'PASS','hidden_dim':config.hidden_dim,'parameters':params,
                      'input_shape':[2,4096],'mu_shape':[2,10],
                      'beta4_loss':float(loss.detach()),'recon':float(recon.detach()),
                      'kl':float(kl.detach()),'data_mapping_roundtrip':True},indent=2))
    if data:
        ds=DSprites(data,cache_dir)
        assert ds.images.shape==(N_IMAGES,64,64)
        b=ds.batch(np.array([0,1,1024]))
        assert b.shape==(3,4096)
        assert set(torch.unique(b).tolist()).issubset({0,1})
        print('DATA PASS: official NPZ, shape, factor indexing, binary samples and memory-mapped cache')


def benchmark(data: Path, cache_dir: Path | None, batch_size: int, max_steps: int,
              warmup: int, measured: int, device: str):
    if not torch.cuda.is_available() and device=='cuda':
        raise RuntimeError('CUDA not available')
    target = torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
    ds=DSprites(data,cache_dir)
    torch.manual_seed(2026)
    model=BetaVAE().to(target).train()
    opt=torch.optim.Adagrad(model.parameters(),lr=0.01)
    if target.type=='cuda':
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    rng=np.random.default_rng(2026)
    # Include real memmap -> CPU gathering -> GPU transfer, backward and Adagrad update.
    t0=time.monotonic()
    for i in range(warmup + measured):
        ids=rng.integers(0,N_IMAGES,size=batch_size)
        x=ds.batch(ids).to(device=target,dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        logits,mu,lv=model(x)
        loss,*_=loss_terms(logits,x,mu,lv,4)
        loss.backward();opt.step()
        if i==warmup-1:
            if target.type=='cuda':torch.cuda.synchronize()
            t0=time.monotonic()
    if target.type=='cuda':torch.cuda.synchronize()
    dt=time.monotonic()-t0
    secs=dt/measured
    t_eval=time.monotonic()
    model.eval()
    with torch.inference_mode():
        for i in range(measured):
            ids=np.arange(i*batch_size,(i+1)*batch_size) % N_IMAGES
            model.encode(ds.batch(ids).to(device=target,dtype=torch.float32))
    if target.type=='cuda':torch.cuda.synchronize()
    eval_secs=(time.monotonic()-t_eval)/measured
    report={'device':str(target),'steps_measured':measured,'batch_size':batch_size,
            'sec_per_step':round(secs,4),'samples_per_sec':round(batch_size/secs,1),
            'estimated_training_minutes_per_beta':round(max_steps*secs/60,1),
            'estimated_both_training_minutes':round(max_steps*secs*2/60,1),
            'estimated_full_dataset_eval_minutes_each':round(math.ceil(N_IMAGES/batch_size)*eval_secs/60,1),
            'estimates_note':'Warm GPU benchmark extrapolation, NOT actual completed training time.',
            'max_gpu_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,1) if target.type=='cuda' else None}
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=None,help='Optional downloaded official .npz for real-data checks')
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--paper-model',action='store_true',help='Smoke test full 1200-wide architecture')
    p.add_argument('--benchmark',action='store_true',help='Real-data GPU benchmark')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--max-steps',type=int,default=12000)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--measured',type=int,default=12)
    p.add_argument('--device',default='auto',choices=['cpu','cuda','auto'])
    a=p.parse_args()
    smoke(a.paper_model,a.data,a.cache_dir)
    if a.benchmark:
        if a.data is None:raise ValueError('--benchmark requires --data')
        benchmark(a.data,a.cache_dir,a.batch_size,a.max_steps,a.warmup,a.measured,a.device)

if __name__=='__main__':
    main()
