"""Native periodic state-only 1D FNO for fixed observation intervals."""
import torch
from torch import nn


class SpectralConv1D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.width, self.modes = width, modes
        self.weight = nn.Parameter(torch.rand(width, width, modes, dtype=torch.cfloat) / (width * width))

    def forward(self, value):
        coeff = torch.fft.rfft(value, dim=-1)
        if coeff.shape[-1] < self.modes:
            raise ValueError('Grid cannot represent the frozen Fourier modes')
        output = torch.zeros_like(coeff)
        output[..., :self.modes] = torch.einsum('bix,iox->box', coeff[..., :self.modes], self.weight)
        return torch.fft.irfft(output, n=value.shape[-1], dim=-1)


class FNO1D(nn.Module):
    """[batch,grid,1] normalized state -> normalized next state, no lag channel.

    Each PDE has one fixed physical observation interval. All three training
    arms use this identical map; this is a new backbone, not the old DUE OSG.
    """
    def __init__(self, width=32, modes=16, depth=4, projection_width=128):
        super().__init__()
        self.width, self.modes, self.depth = width, modes, depth
        self.projection_width = projection_width
        self.lift = nn.Conv1d(1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1D(width, modes) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.project1 = nn.Conv1d(width, projection_width, 1)
        self.project2 = nn.Conv1d(projection_width, 1, 1)
        self.activation = nn.GELU()

    def forward(self, state):
        if state.ndim != 3 or state.shape[-1] != 1:
            raise ValueError('Expected [batch,grid,1] state')
        value = self.lift(state.transpose(1, 2))
        for spectral, local in zip(self.spectral, self.local):
            value = self.activation(spectral(value) + local(value))
        value = self.project2(self.activation(self.project1(value)))
        return value.transpose(1, 2)


def real_parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
