"""Six-run minimalist trainer. Benchmark first. Checkpoints only at epoch boundaries."""
import argparse, copy, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import load_data,loader
from models import create

BATCH={'SB':256,'LB':None}

def eval_model(model,dataset,batch,device,max_examples=None):
    model.eval();total=0;loss=0.;right=0
    with torch.inference_mode():
        for x,y in loader(dataset,batch,device=device):
            if max_examples is not None and total>=max_examples:break
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            logits=model(x);loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
            right+=(logits.argmax(1)==y).sum().item();total+=len(y)
    return {'loss':loss/total,'accuracy':right/total,'n':total}

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def run(a):
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    set_seed(a.seed)
    ds_train,ds_test=load_data(a.panel,a.cifar_root,a.mnist_file)
    batch=256 if a.regime=='SB' else len(ds_train)//10
    if a.batch_override is not None:batch=a.batch_override
    model=create(a.panel).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=a.lr)
    config={'panel':a.panel,'regime':a.regime,'seed':a.seed,'batch':batch,'lr':a.lr,
            'data':'raw/255, no crop, no augmentation','architecture_version':'independent-v1'}
    root=Path(a.output_root)/f'{a.panel}_{a.regime}_seed{a.seed}'
    if a.mode=='train':root.mkdir(parents=True,exist_ok=True)
    ck=root/'last.pt';epoch0=0;history=[];best=float('inf');bad=0
    if a.mode=='train' and a.checkpoint_every < 1:raise ValueError('checkpoint-every must be >= 1')
    if a.mode=='train' and ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=False)
        if state['config']!=config:raise RuntimeError('Checkpoint configuration differs; use separate output root')
        if state.get('finished'):
            print('Already converged:',root);return
        if (root/'endpoint.pt').exists() and state.get('epoch',0)>=a.max_epochs:
            print('Budget exhausted at requested epoch cap. Increase --max-epochs to continue.');return
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        epoch0=state['epoch'];history=state['history'];best=state['best'];bad=state['bad']
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        print('Resuming after epoch',epoch0)
    print('RUN',config,'device',device,'params',sum(p.numel() for p in model.parameters()),flush=True)
    if device=='cuda':print('GPU',torch.cuda.get_device_name(),'allocated',round(torch.cuda.memory_allocated()/1e9,2),'GB',flush=True)
    if a.mode=='benchmark':
        it=iter(loader(ds_train,batch,shuffle=True,device=device,seed=a.seed))
        n=0; t0=time.perf_counter(); fetch=0.;compute=0.
        model.train()
        for j in range(a.bench_steps):
            s=time.perf_counter()
            try:x,y=next(it)
            except StopIteration:it=iter(loader(ds_train,batch,shuffle=True,device=device,seed=a.seed));x,y=next(it)
            fetch+=time.perf_counter()-s
            s=time.perf_counter();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optim.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x),y);loss.backward();optim.step()
            if device=='cuda':torch.cuda.synchronize()
            compute+=time.perf_counter()-s;n+=len(y)
        secs=time.perf_counter()-t0
        print(json.dumps({'panel':a.panel,'regime':a.regime,'steps':a.bench_steps,'samples':n,'seconds':secs,
              'samples_per_sec':n/secs,'batch_fetch_seconds':fetch,'train_step_seconds':compute,
              'estimated_epoch_seconds_at_constant_throughput':len(ds_train)/n*secs,
              'gpu_peak_gb':round(torch.cuda.max_memory_allocated()/1e9,3) if device=='cuda' else None},indent=2))
        return
    if a.mode=='smoke':
        x,y=next(iter(loader(ds_train,min(batch,32),device=device)))
        optim.zero_grad(set_to_none=True);out=model(x.to(device));loss=nn.functional.cross_entropy(out,y.to(device));loss.backward();optim.step()
        print('smoke loss',loss.item(),'shape',tuple(out.shape));return
    stop_reason='budget_exhausted'
    start=time.monotonic()
    epoch=epoch0
    for epoch in range(epoch0+1,a.max_epochs+1):
        model.train();running=torch.zeros((),device=device);count=0
        # Epoch-seeded permutation makes epoch-boundary resumes deterministic.
        for x,y in loader(ds_train,batch,shuffle=True,device=device,seed=a.seed+epoch):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            loss.backward();optim.step()
            running+=loss.detach()*len(y);count+=len(y)
        rec={'epoch':epoch,'training_batch_loss':(running/count).item(),'elapsed_s':time.monotonic()-start}
        if epoch%a.eval_every==0 or epoch==a.max_epochs:
            # Evaluations on deterministic fixed prefixes; full passes ONLY at finalization.
            rec['train_probe']=eval_model(model,ds_train,a.eval_batch,device,a.probe_train)
            rec['test_probe']=eval_model(model,ds_test,a.eval_batch,device,a.probe_test)
            probe=rec['train_probe']['loss']
            if probe<best-a.min_delta:best=probe;bad=0
            else:bad+=1
            print(f'{a.panel}_{a.regime} epoch {epoch}: train_step_loss={rec["training_batch_loss"]:.4f} probe_loss={probe:.4f} train_acc={rec["train_probe"]["accuracy"]:.3f} test_acc={rec["test_probe"]["accuracy"]:.3f} bad={bad}',flush=True)
        history.append(rec)
        converged=(epoch>=a.min_epochs and bad>=a.patience)
        finished=converged
        should_save=(epoch % a.checkpoint_every == 0 or converged or epoch==a.max_epochs)
        if should_save:
            save={'config':config,'epoch':epoch,'history':history,'best':best,'bad':bad,
                  'model':model.state_dict(),'optimizer':optim.state_dict(),
                  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                  'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'finished':finished}
            tmp=ck.with_suffix('.tmp');torch.save(save,tmp);os.replace(tmp,ck)
        if finished:
            stop_reason='loss_plateau' if converged else 'budget_exhausted';break
    final={'config':config,'epoch':epoch,'stop_reason':stop_reason,'elapsed_s':time.monotonic()-start,
           'train':eval_model(model,ds_train,a.eval_batch,device),'test':eval_model(model,ds_test,a.eval_batch,device),
           'history':history,'batchnorm_policy':'trained running stats; interpolation separately recalibrates'}
    (root/'metrics.json').write_text(json.dumps(final,indent=2))
    print('FINAL',json.dumps({k:v for k,v in final.items() if k!='history'},indent=2))
    # Explicitly exclude optimizer from compact inference checkpoint.
    torch.save({'config':config,'model':{k:v.cpu() for k,v in model.state_dict().items()},'final':final},root/'endpoint.pt')

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['benchmark','smoke','train'],default='benchmark')
    p.add_argument('--panel',choices=['F1','C1','C2'],required=True)
    p.add_argument('--regime',choices=['SB','LB'],required=True)
    p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cifar-root');p.add_argument('--mnist-file');p.add_argument('--output-root',default='results')
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--bench-steps',type=int,default=8)
    p.add_argument('--max-epochs',type=int,default=40);p.add_argument('--min-epochs',type=int,default=15)
    p.add_argument('--eval-every',type=int,default=5);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--min-delta',type=float,default=0.002)
    p.add_argument('--probe-train',type=int,default=2048);p.add_argument('--probe-test',type=int,default=2048)
    p.add_argument('--eval-batch',type=int,default=512)
    p.add_argument('--checkpoint-every',type=int,default=5)
    p.add_argument('--batch-override',type=int,help='DIAGNOSTIC ONLY; changes paper batch size')
    return p.parse_args()
if __name__=='__main__':run(parse())
