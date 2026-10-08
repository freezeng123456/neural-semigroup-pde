#!/usr/bin/env python3
"""Run unmodified DUE osgnet definitions on synthetic scalar exponential decay.

This is a software smoke check, not a reproduction of a published benchmark.
An AST loader selects original definitions without rewriting their bodies,
avoiding unrelated scipy/matplotlib imports absent from the current runtime.
It does not test DUE's complete package import, data loader, or trainer.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

PINNED_COMMIT = "1e629104a12a9c3e3064996cbd4e301cec641ad8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_definition(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(selected) != 1:
        raise ValueError(f"exactly one definition required: {name}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return {"path": str(path), "sha256": sha(path), "definition": name,
            "line_start": selected[0].lineno, "line_end": selected[0].end_lineno}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--due-dir", type=Path, default=Path(__file__).parent / "due")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "due_scalar_smoke.json")
    args = parser.parse_args()
    root = args.due_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("DUE commit differs from the locked source")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
        raise ValueError("DUE tracked source has local modifications")
    namespace = {"torch": torch, "np": np, "os": os}
    source = [load_definition(root / "due/utils.py", "get_activation", namespace),
              load_definition(root / "due/networks/nn.py", "nn", namespace),
              load_definition(root / "due/networks/fcn.py", "osgnet", namespace)]
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    model = namespace["osgnet"](
        np.array([[-1.]]), np.array([[1.]]), .025, .1,
        {"dtype": "double", "problem_dim": 1, "depth": 1, "width": 8,
         "activation": "tanh", "seed": 31415}, multiscale=False)

    def data(count, seed):
        rng = torch.Generator().manual_seed(seed)
        state = 2*torch.rand(count, 1, generator=rng)-1
        lag = .025+.075*torch.rand(count, 1, generator=rng)
        return state, lag, state*torch.exp(-lag)

    def forward(state, lag):
        time_feature = 2*(lag-.5*(.1+.025))/(.1-.025)
        return model(torch.cat((state, time_feature), dim=-1))

    train, val = data(128, 2026100401), data(64, 2026100402)
    collocation = torch.Generator().manual_seed(2026100403)
    random_state = 2*torch.rand(128, 1, generator=collocation)-1
    # Match DUE's default independent sublag sampling for this author smoke.
    s = .025+.075*torch.rand(128, 1, generator=collocation)
    t = .025+.075*torch.rand(128, 1, generator=collocation)

    def losses():
        prediction = forward(train[0], train[1])
        fit = (prediction-train[2]).square().mean()
        direct = forward(random_state, s+t)
        st = forward(forward(random_state, s), t)
        ts = forward(forward(random_state, t), s)
        sg = .5*((st-direct).square().mean()+(ts-direct).square().mean())
        return fit, sg, .5*(fit+sg)

    def measures():
        with torch.no_grad():
            fit, sg, total = losses()
            validation = (forward(val[0], val[1])-val[2]).square().mean()
            identity = (forward(val[0], torch.zeros_like(val[1]))-val[0]).abs().max()
            return {"train_mse": fit.item(), "gdsg_mse": sg.item(),
                    "objective": total.item(), "validation_mse": validation.item(),
                    "identity_max_abs_error": identity.item()}

    before = measures()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    start = time.perf_counter()
    finite_gradients = True
    history = []
    for step in range(1, 501):
        fit, sg, total = losses()
        optimizer.zero_grad()
        total.backward()
        finite_gradients &= bool(torch.isfinite(total)) and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters())
        if not finite_gradients:
            raise FloatingPointError("non-finite loss or gradient in author smoke")
        optimizer.step()
        if step in (1, 100, 250, 500):
            history.append({"step": step, **measures()})
    after = measures()
    checks = {"training_improved": after["train_mse"] < before["train_mse"],
              "validation_improved": after["validation_mse"] < before["validation_mse"],
              "identity_within_roundoff": after["identity_max_abs_error"] < 1e-14,
              "all_losses_and_gradients_finite": finite_gradients}
    result = {"schema_version": "due-author-scalar-software-smoke-v1",
              "scope": "Synthetic software check; not a published benchmark reproduction.",
              "date": "2026-10-04", "repository": "https://github.com/AI4Equations/due",
              "commit": commit, "license": "GNU LGPL v2.1",
              "license_sha256": sha(root / "LICENSE"), "source_definitions": source,
              "loader": "Compile original AST definitions unchanged; skip package-level plotting/I/O imports.",
              "not_tested": ["full due package import", "DUE data loaders", "DUE ODE_osg trainer", "published PDE benchmarks", "GPU execution"],
              "script_sha256": sha(Path(__file__).resolve()),
              "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
              "device": "cpu", "threads": 1,
              "equation": "du/dt=-u, exact target u(t)=u(0)*exp(-t)",
              "configuration": {"seed": 31415, "n_train":128, "n_validation":64,
                                "hidden":8, "depth":1, "parameters":sum(p.numel() for p in model.parameters()),
                                "training_lag_range":[.025,.1], "sg_sublag_range":[.025,.1],
                                "sg_sum_can_exceed_training_support":True,
                                "steps":500, "optimizer":"Adam", "learning_rate":.003,
                                "sg_weight":1., "objective":"(fit + sg)/2"},
              "before":before, "after":after, "history":history,
              "elapsed_seconds":time.perf_counter()-start,
              "checks":checks, "passed":all(checks.values())}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed":result["passed"], "before":before, "after":after, "output":str(args.output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
