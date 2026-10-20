"""Measured linear parameter interpolation; no training or hardcoded metrics."""
import argparse,json,copy,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import load_data,loader
from models import create
from train import eval_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panel',choices=['F1','C1','C2'],required=True)
    p.add_argument('--output-root',default='results');p.add_argument('--device',default='auto')
    p.add_argument('--cifar-root');p.add_argument('--mnist-file')
    p.add_argument('--points',type=int,default=21);p.add_argument('--eval-batch',type=int,default=512)
    p.add_argument('--bn-policy',choices=['interpolate','recalibrate'],default='recalibrate')
    p.add_argument('--bn-examples',type=int,default=4096,help='fixed training prefix used for BN recalibration')
    a=p.parse_args();dev='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    roots=[Path(a.output_root)/f'{a.panel}_{regime}_seed42' for regime in ('SB','LB')]
    saved=[torch.load(r/'endpoint.pt',map_location='cpu',weights_only=False) for r in roots]
    assert saved[0]['config']['panel']==saved[1]['config']['panel']==a.panel
    assert saved[0]['config']['seed']==saved[1]['config']['seed']==42
    a_state,b_state=[z['model'] for z in saved]
    assert a_state.keys()==b_state.keys()
    train,test=load_data(a.panel,a.cifar_root,a.mnist_file)
    model=create(a.panel).to(dev)
    # Exact endpoints under original running statistics (separate from curve policy).
    endpoints=[]
    for state in (a_state,b_state):
        model.load_state_dict(state)
        endpoints.append({'train':eval_model(model,train,a.eval_batch,dev),'test':eval_model(model,test,a.eval_batch,dev)})
    results=[];alpha_vals=np.linspace(-1,2,a.points)
    for alpha in alpha_vals:
        state={}
        for key,u in a_state.items():
            v=b_state[key]
            # BN integer buffers must not be interpolated.
            state[key]=torch.lerp(u.float(),v.float(),float(alpha)).to(u.dtype) if u.is_floating_point() else u.clone()
        model.load_state_dict(state)
        if a.bn_policy=='recalibrate':
            # Forward-only update of BN buffers; no gradients and no optimizer.
            for m in model.modules():
                if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                    m.reset_running_stats();m.momentum=None;m.train()
            model.train() # BN updates; dropout is then individually disabled.
            for m in model.modules():
                if isinstance(m,nn.Dropout):m.eval()
            seen=0
            with torch.no_grad():
                for x,y in loader(train,min(a.eval_batch,256),device=dev):
                    if seen>=a.bn_examples:break
                    model(x.to(dev,non_blocking=True));seen+=len(x)
        tr=eval_model(model,train,a.eval_batch,dev)
        te=eval_model(model,test,a.eval_batch,dev)
        results.append({'alpha':float(alpha),'train_loss':tr['loss'],'test_loss':te['loss'],
                        'train_accuracy':tr['accuracy'],'test_accuracy':te['accuracy']})
        print(f'{a.panel} alpha={alpha:+.3f} trainCE={tr["loss"]:.4f} testCE={te["loss"]:.4f}',flush=True)
    out=Path(a.output_root)/f'{a.panel}_interpolation.json'
    out.write_text(json.dumps({'panel':a.panel,'method':'linear trainable parameters; no gradients',
              'bn_policy':a.bn_policy,'bn_examples':a.bn_examples if a.bn_policy=='recalibrate' else None,
              'endpoint_original_bn':endpoints,'points':results},indent=2))
    print('Saved:',out)
if __name__=='__main__':main()
