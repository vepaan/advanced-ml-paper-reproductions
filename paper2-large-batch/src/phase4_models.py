"""Phase 4 C1/C2 variants based on Keskar et al. authors' network_zoo.py.
Separate from Phase 3 models.py, which must remain unchanged.
https://github.com/keskarnitish/large-batch-training/blob/master/network_zoo.py
"""
import torch
from torch import nn
from models import F1  # Unmodified Phase 3 F1: no author F1 implementation in referenced code.

VERSION = 'phase4-author-keras-architecture-v1'


def conv(cin, cout, kernel, stride=1, padding=None):
    if padding is None:
        padding = kernel // 2
    return [nn.Conv2d(cin, cout, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def dense(cin, cout, dropout=0.0):
    result = [nn.Linear(cin, cout, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if dropout:
        result.append(nn.Dropout(dropout))
    return result


class C1(nn.Module):
    """Author shallownet: two 5x5 stride-1 SAME conv + 3x3 stride-2 SAME pool."""
    def __init__(self):
        super().__init__()
        layers = []
        prev = 3
        for _ in range(2):
            layers.extend(conv(prev, 64, kernel=5, stride=1, padding=2))
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
            prev = 64
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        *dense(64 * 8 * 8, 384, dropout=0.5),
                                        *dense(384, 192, dropout=0.5),
                                        nn.Linear(192, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class C2(nn.Module):
    """Author deepnet dropout placements; Phase 3 conv/BN bias convention retained."""
    def __init__(self):
        super().__init__()
        layers = []
        prev = 3
        # Original deepnet(): rates after nonfinal convolutions of each block.
        for channels, rates in [(64, [.3, 0]),
                                (128, [.4, 0]),
                                (256, [.4, .4, 0]),
                                (512, [.4, .4, 0]),
                                (512, [.4, .4, 0])]:
            for rate in rates:
                layers.extend(conv(prev, channels, kernel=3))
                if rate:
                    layers.append(nn.Dropout(rate))
                prev = channels
            layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.5),
                                        *dense(512, 512, dropout=0.5),
                                        nn.Linear(512, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def create(panel):
    return {'F1': F1, 'C1': C1, 'C2': C2}[panel]()


def audit(panel, device='cpu'):
    m = create(panel).to(device)
    m.train()
    shape = (2, 1, 28, 28) if panel == 'F1' else (2, 3, 32, 32)
    out = m(torch.randn(shape, device=device))
    assert out.shape == (2, 10)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    n = sum(p.numel() for p in m.parameters())
    print(f'{panel}: parameters={n:,}, forward/backward OK')
    return n


if __name__ == '__main__':
    for p in ('C1', 'C2'):
        audit(p)
