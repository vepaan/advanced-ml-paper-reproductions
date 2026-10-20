"""Independently derived from Keskar et al. Appendix B; no imported architectures."""
import torch
from torch import nn


def conv(cin,cout,k=3,stride=1,padding=1):
    return [nn.Conv2d(cin,cout,k,stride,padding,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True)]

def dense(cin,cout,dropout=0):
    out=[nn.Linear(cin,cout,bias=False),nn.BatchNorm1d(cout),nn.ReLU(inplace=True)]
    if dropout: out.append(nn.Dropout(dropout))
    return out

class F1(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[nn.Flatten()]; prev=784
        for _ in range(5):
            blocks.extend(dense(prev,512));prev=512
        blocks.append(nn.Linear(512,10));self.net=nn.Sequential(*blocks)
    def forward(self,x): return self.net(x)

class C1(nn.Module):
    def __init__(self):
        super().__init__()
        # Paper does not specify padding/pooling stride. SAME-like conv pad=2,
        # pool(kernel=3,stride=2,pad=1): 32->16->8->4->2.
        stages=[]; prev=3
        for _ in range(2):
            stages.extend(conv(prev,64,5,2,2)); stages.append(nn.MaxPool2d(3,2,1));prev=64
        self.features=nn.Sequential(*stages)
        self.classifier=nn.Sequential(nn.Flatten(),*dense(64*2*2,384,0.5),
                                      *dense(384,192,0.5),nn.Linear(192,10))
    def forward(self,x): return self.classifier(self.features(x))

class C2(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];prev=3
        for count,ch in ((2,64),(2,128),(3,256),(3,512),(3,512)):
            for _ in range(count): layers.extend(conv(prev,ch));prev=ch
            layers.append(nn.MaxPool2d(2,2))
        self.features=nn.Sequential(*layers)
        # B.4 says properties as B.3, interpreted as 50% dense dropout.
        self.classifier=nn.Sequential(nn.Flatten(),*dense(512,512,0.5),nn.Linear(512,10))
    def forward(self,x):return self.classifier(self.features(x))

def create(panel): return {'F1':F1,'C1':C1,'C2':C2}[panel]()

def audit(panel,device='cpu'):
    m=create(panel).to(device)
    shape=(2,1,28,28) if panel=='F1' else (2,3,32,32)
    x=torch.randn(shape,device=device); y=m(x)
    assert y.shape==(2,10)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print(panel,'parameters',sum(p.numel() for p in m.parameters()),'logits',tuple(y.shape),'forward/backward OK')

if __name__=='__main__':
    for panel in ['F1','C1','C2']: audit(panel)
