"""Higgins et al. (2017) Figure 7A/B: position, scale, rotation, traversals.

No prebuilt target values. Everything comes from locally trained model states.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .data import DSprites, DEFAULT_DATA, index_from_factors
from .evaluate import load_inference, calculate_factor_responses, reconstruction_diagnostics


@torch.inference_mode()
def traverse(model, ds, device, latent_indices, reference_factors=(0,1,3,10,16,16)):
    # Paper: vary 1 unit from +3 to -3; all 9 remaining values inferred from
    # SAME dataset reference image, not all zeros. Use mean for reproducibility.
    idx=int(index_from_factors(np.array(reference_factors)))
    x=ds.batch(np.array([idx])).to(device=device,dtype=torch.float32)
    mu,_=model.encode(x)
    images=[]
    for value in np.linspace(3.0,-3.0,5):
        z=mu.repeat(len(latent_indices),1)
        for row,j in enumerate(latent_indices):
            z[row,j]=float(value)
        recon=torch.sigmoid(model.decode_logits(z)).reshape(-1,64,64)
        images.append(recon.cpu().numpy())
    return np.stack(images)  # [5, 6, 64,64]


def _scalar(v):
    return float(v) if hasattr(v,'item') else v


def data_for_panel(ckpt_path, ds, device, eval_batch_size):
    model,meta=load_inference(ckpt_path,device)
    factors=calculate_factor_responses(model,ds,device,eval_batch_size)
    sorted_idxs=list(map(int,factors['sorted_latents']))
    selected=sorted_idxs[:5]+[sorted_idxs[-1]]  # five most active + most inactive
    generated=traverse(model,ds,device,selected)
    diagnostics=reconstruction_diagnostics(model,ds,device)
    return {'meta':meta,'factors':factors,'selected':selected,
            'generated':generated,'diagnostics':diagnostics}


def draw_panel(panel, ax, label):
    # ax: [8,6] (position, scale, rotation, then 5 traversal rows).
    response=panel['factors']; dims=panel['selected']; ims=panel['generated']
    pvar=response['posterior_variance']
    cmap='jet'  # original high=red; each latent has own heatmap color limits
    for c,dim in enumerate(dims):
        heat=ax[0,c]
        heat.imshow(response['position_mean'][:,:,dim],origin='lower',cmap=cmap,interpolation='nearest')
        heat.set_title(f'z{dim+1}' if c<5 else f'z{dim+1} (inactive)',fontsize=9)
        for spine in heat.spines.values(): spine.set_visible(False)
        heat.set_xticks([]);heat.set_yticks([])
        for r, array, kind in [(1,response['scale_mean'],'scale'),(2,response['rotation_mean'],'rotation')]:
            axis=ax[r,c]
            for shape,color in enumerate(('red','green','blue')):
                axis.plot(array[shape,:,dim],color=color,linewidth=.85)
            axis.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            axis.grid(False)
        for r in range(5):
            axis=ax[3+r,c]
            axis.imshow(ims[r,c],vmin=0,vmax=1,cmap='gray',interpolation='nearest')
            axis.set_xticks([]); axis.set_yticks([])
        ax[7,c].set_xlabel(f'{pvar[dim]:.2f}',fontsize=8)
    for r, name in enumerate(('position','scale','rotation','+3','+1.5','0','-1.5','-3')):
        ax[r,0].set_ylabel(name,fontsize=8,rotation=90,labelpad=6)
    title= 'A: β-VAE (β=4)' if label=='A' else 'B: VAE (β=1)'
    ax[0,0].text(0,1.65,title,transform=ax[0,0].transAxes,fontweight='bold',fontsize=14)


def make_one(panel, label, dest:Path):
    fig,axes=plt.subplots(8,6,figsize=(10.2,14),gridspec_kw={'height_ratios':[1.05,1.05,1.05,1,1,1,1,1]})
    draw_panel(panel,axes,label)
    fig.subplots_adjust(wspace=.07,hspace=.10,left=.08,right=.99,top=.95,bottom=.07)
    fig.savefig(dest,dpi=150,bbox_inches='tight'); plt.close(fig)


def make_joint(a,b,dest:Path):
    fig,axes=plt.subplots(8,12,figsize=(20,14),gridspec_kw={'height_ratios':[1.05,1.05,1.05,1,1,1,1,1]})
    draw_panel(a,axes[:,:6],'A');draw_panel(b,axes[:,6:],'B')
    for r in range(8): axes[r,6].set_ylabel('')
    fig.subplots_adjust(wspace=.06,hspace=.10,left=.04,right=.995,top=.95,bottom=.07)
    fig.savefig(dest,dpi=150,bbox_inches='tight'); plt.close(fig)


def _report(panel):
    f=panel['factors'];m=panel['meta']
    return {'beta':m['beta'],'seed':m['seed'],'steps':m['step'],
            'mean_posterior_var_all_10':list(map(float,f['posterior_variance'])),
            'mean_kl_all_10':list(map(float,f['kl_per_latent'])),
            'ordering_by_ascending_mean_posterior_variance_0_based':list(map(int,f['sorted_latents'])),
            'displayed_latent_ids_1_based':[x+1 for x in panel['selected']],
            'reconstruction':panel['diagnostics']}


def run(args):
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    ds=DSprites(args.data,args.cache_dir)
    args.figures_dir.mkdir(parents=True,exist_ok=True)
    output={}
    a,b=None,None
    if args.panel in ('both','A'):
        a=data_for_panel(args.output_root/'beta4'/'model.pt',ds,device,args.eval_batch_size)
        make_one(a,'A',args.figures_dir/'figure7_A.png')
        output['A']=_report(a)
        print('Figure 7A ready',output['A'],flush=True)
    if args.panel in ('both','B'):
        b=data_for_panel(args.output_root/'beta1'/'model.pt',ds,device,args.eval_batch_size)
        make_one(b,'B',args.figures_dir/'figure7_B.png')
        output['B']=_report(b)
        print('Figure 7B ready',output['B'],flush=True)
    if args.panel=='both':
        make_joint(a,b,args.figures_dir/'figure7_AB.png')
    with open(args.figures_dir/'figure7_metrics.json','w') as out:
        json.dump(output,out,indent=2)
    print('Saved all outputs to:',args.figures_dir)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--panel', choices=['A','B','both'],default='both')
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('/content/paper3_runs' if Path('/content').exists() else './results'))
    p.add_argument('--figures-dir',type=Path,default=Path('./figures'))
    p.add_argument('--eval-batch-size',type=int,default=2048)
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    return p.parse_args()


if __name__=='__main__':
    run(arguments())
