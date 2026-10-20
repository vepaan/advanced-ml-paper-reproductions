"""Paper 2 raw MNIST/CIFAR-10; no Paper 1 crop or whitening."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

def load_data(panel, cifar_root=None, mnist_file=None):
    if panel == 'F1':
        path = Path(mnist_file) if mnist_file else ROOT / 'data' / 'mnist.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; download mnist.npz using README instructions')
        with np.load(path, allow_pickle=False) as d:
            a,b,c,e = (d[k] for k in ('x_train','y_train','x_test','y_test'))
        xtr = torch.from_numpy(a.copy()).unsqueeze(1).float().div_(255)
        xte = torch.from_numpy(c.copy()).unsqueeze(1).float().div_(255)
        ytr,yte = torch.from_numpy(b.astype(np.int64)),torch.from_numpy(e.astype(np.int64))
    else:
        root = Path(cifar_root) if cifar_root else REPO / 'paper1-generalization' / 'data'
        if (root / 'cifar-10-batches-py').exists(): root = root / 'cifar-10-batches-py'
        def read(names):
            xs,ys=[],[]
            for name in names:
                with (root/name).open('rb') as f: obj=pickle.load(f, encoding='bytes')
                xs.append(obj[b'data']); ys.extend(obj[b'labels'])
            arr=np.concatenate(xs).reshape(-1,3,32,32)
            return torch.from_numpy(arr.copy()).float().div_(255),torch.tensor(ys,dtype=torch.long)
        xtr,ytr=read([f'data_batch_{i}' for i in range(1,6)])
        xte,yte=read(['test_batch'])
    assert len(xtr)==(60000 if panel=='F1' else 50000) and len(xte)==10000
    assert 0<=xtr.min() and xtr.max()<=1 and ytr.min()>=0 and ytr.max()<10
    return TensorDataset(xtr,ytr),TensorDataset(xte,yte)

def loader(ds,batch,shuffle=False,device='cpu',seed=42):
    # Data already decoded into contiguous RAM tensors; extra workers duplicate overhead.
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,
                      pin_memory=str(device).startswith('cuda'),
                      generator=torch.Generator().manual_seed(seed))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--panel',choices=['F1','C1','C2'],required=True)
    p.add_argument('--cifar-root');p.add_argument('--mnist-file');args=p.parse_args()
    train,test=load_data(args.panel,args.cifar_root,args.mnist_file)
    for name,ds in [('train',train),('test',test)]:
        x,y=ds.tensors
        print(args.panel,name,tuple(x.shape),tuple(y.shape),'min/max',float(x.min()),float(x.max()),'classes',sorted(y.unique().tolist()))
