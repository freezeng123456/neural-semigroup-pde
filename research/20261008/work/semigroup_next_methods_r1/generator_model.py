"""DUE's unchanged spatial FNO backbone as a velocity field.

The autonomous arm fixes the time feature to zero. Both arms allocate exactly
the same parameters and share initialization. Twenty input weights are inactive
in the autonomous arm; this is reported rather than called exact effective
capacity equality. The forward output is dz/d physical time, without a residual
state addition or a lag multiplier.
"""
import numpy as np
import torch
from due.networks.fno import osg_fno2d
from flow_numerics import rk4


class GeneratorFNO(osg_fno2d):
    def __init__(self, vmin, vmax, config, autonomous):
        super().__init__(vmin, vmax, 0., 75., config, multiscale=False)
        self.autonomous = bool(autonomous)

    def field(self, state, physical_time):
        if physical_time.shape != (state.shape[0],):
            raise ValueError('Expected one absolute physical time per sample')
        feature = torch.zeros_like(physical_time) if self.autonomous else 2*physical_time/75.-1
        feature = feature[:, None, None, None].expand(*state.shape[:-1], 1)
        return self(torch.cat((state, feature), dim=-1))

    def forward(self, x):
        x = self.en(x).permute(0, 3, 1, 2)
        for i in range(self.nblocks):
            x = self.activation(self.mlp[i](self.conv[i](x))+self.w[i](x))
        return self.de(x).permute(0, 2, 3, 1)

    def predict(self, x, dt, device, substeps=2, start_time=None):
        self.to(device); self.eval()
        n, h, w, c = x.shape
        if c != 1 or dt.shape[0] != n or not np.isfinite(x).all() or not (dt > 0).all():
            raise ValueError('Invalid rollout interface')
        lo, hi = self.vmin[..., 0].to(device), self.vmax[..., 0].to(device)
        z = (2*(torch.from_numpy(x).to(device)-.5*(lo+hi))/(hi-lo)).float()
        lags = torch.from_numpy(dt).to(device).float()
        now = torch.zeros(n, device=device) if start_time is None else torch.as_tensor(start_time, device=device, dtype=torch.float32)
        prediction = np.full((n, h, w, 1, dt.shape[1]+1), np.nan, dtype=np.float32)
        prediction[..., 0] = (z*.5*(hi-lo)+.5*(hi+lo)).cpu().numpy()
        with torch.no_grad():
            for k in range(dt.shape[1]):
                z = rk4(self.field, z, now, lags[:, k], substeps)
                prediction[..., k+1] = (z*.5*(hi-lo)+.5*(hi+lo)).cpu().numpy()
                now = now+lags[:, k]
                if not torch.isfinite(z).all():
                    break  # Preserve the finite prefix and non-finite failure.
        return prediction
