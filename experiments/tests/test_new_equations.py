#!/usr/bin/env python3
"""Analytic conservation/CFL and independent linear-wave checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import run_new_equations as d

def main():
 torch.set_num_threads(1)
 assert len(d.matrix())==24
 for mode in ('autonomous','query'):
  torch.manual_seed(20260910)
  m=d.Flow('diffusion',mode).double();y=d.states(4,390909,'diffusion',True);z=m(y,.24,2.)
  assert (z.mean(-1)-y.mean(-1)).abs().max()<1e-12
  assert (z.amin(-1)>=y.amin(-1)-1e-12).all() and (z.amax(-1)<=y.amax(-1)+1e-12).all()
  assert (d.energy(z,'diffusion',2.)-d.energy(y,'diffusion',2.)).max()<1e-12
  assert torch.equal(m(torch.ones(2,d.N,dtype=torch.float64)*.3,.24,2.),torch.ones(2,d.N,dtype=torch.float64)*.3)
  z.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters())
 m=d.Flow('sine_gordon','query').double();y=d.states(4,390909,'sine_gordon',True);ref=y.clone();h=.0001
 def rhs(a):return torch.stack((a[:,1],1.5**2*d.lap(a[:,0])-.15*a[:,1]),1)
 for _ in range(1200):
  a=rhs(ref);b=rhs(ref+h*a/2);c=rhs(ref+h*b/2);e=rhs(ref+h*c);ref=ref+h*(a+2*b+2*c+e)/6
 linear=m.wave_linear(y,d.duration(.12,y),1.5)
 assert (linear-ref).abs().max()<1e-9
 z=m(y,.12,1.5);z.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters())
 assert torch.isfinite(z).all()
 print('PASS: paired matrix; diffusion mass, bounds, energy and stationary states; independent RK4 linear-wave agreement; finite gradients')
if __name__=='__main__':main()
