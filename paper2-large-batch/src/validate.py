"""Fast architecture/data audit without full training."""
import argparse
from models import audit
from data import load_data
p=argparse.ArgumentParser();p.add_argument('--panel',choices=['F1','C1','C2','all'],default='all');p.add_argument('--cifar-root');p.add_argument('--mnist-file');p.add_argument('--data',action='store_true');a=p.parse_args()
for panel in (['F1','C1','C2'] if a.panel=='all' else [a.panel]):
    audit(panel)
    if a.data:
        tr,te=load_data(panel,a.cifar_root,a.mnist_file)
        print('  data',len(tr),len(te),'shape',tuple(tr.tensors[0].shape[1:]))
