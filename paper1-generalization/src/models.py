"""Independent CIFAR-10 model implementations for Zhang et al. (2017), Fig. 1.

Figure 3 and Appendix A specify the major architecture. Padding, convolution
biases, AlexNet filter widths, and LRN settings require documented choices.
No pretrained weights or third-party model implementations are used.
"""
from __future__ import annotations

import argparse
import torch
from torch import nn
import torch.nn.functional as F


class ConvModule(nn.Sequential):
    """Convolution -> BatchNorm -> ReLU (Figure 3)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class InceptionModule(nn.Module):
    """Parallel 1x1 and 3x3 convolution branches, concatenated in channels."""
    def __init__(self, in_channels: int, ch1: int, ch3: int):
        super().__init__()
        self.branch1 = ConvModule(in_channels, ch1, 1)
        self.branch3 = ConvModule(in_channels, ch3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch1(x), self.branch3(x)), dim=1)


class DownsampleModule(nn.Module):
    """Stride-2 3x3 conv branch and stride-2 3x3 max-pool branch."""
    def __init__(self, in_channels: int, conv_channels: int):
        super().__init__()
        self.conv = ConvModule(in_channels, conv_channels, 3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.conv(x), self.pool(x)), dim=1)


class SmallInception(nn.Module):
    """Figure 3 sequence, bottom-to-top within each diagram column."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvModule(3, 96, 3),
            InceptionModule(96, 32, 48),
            InceptionModule(80, 32, 32),
            DownsampleModule(64, 80),
            InceptionModule(144, 48, 96),
            InceptionModule(144, 80, 80),
            InceptionModule(160, 96, 64),
            InceptionModule(160, 112, 48),
            DownsampleModule(160, 96),
            InceptionModule(256, 176, 160),
            InceptionModule(336, 176, 160),
        )
        self.classifier = nn.Linear(336, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


class AdaptedAlexNet(nn.Module):
    """Appendix A's two conv/pool/LRN stages and 384/192-unit dense layers.

    Both convolutions use 64 channels and same-padding 5x5 kernels; pooling
    uses 3x3, stride 2, padding 1. This candidate matches the paper's
    parameter count, but unspecified details remain reconstruction choices.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1), nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1), nn.LocalResponseNorm(5),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 3, 28, 28)).numel()
        self.classifier = nn.Sequential(
            nn.Linear(flat, 384), nn.ReLU(inplace=True),
            nn.Linear(384, 192), nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class MLP1x512(nn.Module):
    """One ReLU hidden layer of width 512, with a 10-way linear output."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 28 * 28, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


MODELS = {"inception": SmallInception, "alexnet": AdaptedAlexNet,
          "mlp": MLP1x512}
PAPER_PARAMETER_COUNTS = {"inception": 1_649_402,
                          "alexnet": 1_387_786, "mlp": 1_209_866}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def audit_layers(model: nn.Module, name: str) -> None:
    """Report each leaf layer's parameter count and output shape on 28x28 input.

    Forward hooks observe the actual graph without changing model behavior.
    A batch of two supports BatchNorm in training mode; evaluation mode
    avoids altering running statistics during the audit.
    """
    records: list[tuple[str, str, tuple[int, ...], int]] = []
    hooks = []
    for path, layer in model.named_modules():
        if not path or any(layer.children()):
            continue

        def capture(module: nn.Module, _inputs: tuple, output: torch.Tensor,
                    path: str = path) -> None:
            parameters = sum(p.numel() for p in module.parameters(recurse=False))
            records.append((path, module.__class__.__name__, tuple(output.shape), parameters))

        hooks.append(layer.register_forward_hook(capture))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model(torch.zeros(2, 3, 28, 28))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    print(f"\n{name.upper()} LAYER AUDIT (input=(2,3,28,28))")
    print(f"{'layer':<38} {'type':<19} {'output shape':<20} {'parameters':>12}")
    print('-' * 96)
    total = 0
    for path, layer_type, shape, parameters in records:
        total += parameters
        print(f"{path:<38} {layer_type:<19} {str(shape):<20} {parameters:>12,}")
    expected = count_parameters(model)
    assert total == expected, (name, total, expected)
    print(f"Total: {total:,}; paper: {PAPER_PARAMETER_COUNTS[name]:,}; "
          f"difference: {total - PAPER_PARAMETER_COUNTS[name]:+,}; "
          f"final output: {tuple(result.shape)}")


def smoke_test() -> None:
    torch.manual_seed(42)
    for name, factory in MODELS.items():
        model = factory()
        model.train()
        x = torch.randn(2, 3, 28, 28)
        logits = model(x)
        assert logits.shape == (2, 10), (name, logits.shape)
        assert torch.isfinite(logits).all(), name
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad), name
        actual = count_parameters(model)
        expected = PAPER_PARAMETER_COUNTS[name]
        print(f"{name:>10}: output={tuple(logits.shape)}, "
              f"parameters={actual:,}, paper={expected:,}, "
              f"difference={actual - expected:+,}, loss={loss.item():.4f}")
    print("PASS: forward/backward/finite-gradient smoke tests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-layers", action="store_true")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()
    if args.smoke_test or not args.audit_layers:
        smoke_test()
    if args.audit_layers:
        names = MODELS if args.model == "all" else [args.model]
        for name in names:
            audit_layers(MODELS[name](), name)
