#!/usr/bin/env python3
"""Re-evaluate recovered legacy checkpoints; no optimization is performed."""
import argparse
import json
from pathlib import Path

import torch
import run_generator_coverage as current
from run_generator_coverage import base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.legacy / "cache_metadata.json").read_text())
    assert base.digest(args.legacy / "cache.pt") == metadata["sha256"]
    data = base.device_tree(torch.load(args.legacy / "cache.pt", map_location="cpu", weights_only=False), args.device)
    results = {}
    for cell in sorted((args.legacy / "cells").glob("*-n128-k4-*")):
        config = json.loads((cell / "config.json").read_text())
        model = base.ReactionFlow(config["conditioned"], 4, True).to(args.device)
        train = data["train"][config["seed"]]
        train = {**train, "states": torch.stack((train["u"], train["target"]), 1)}
        for name in ("best", "final"):
            path = cell / (name + ".pt")
            model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
            result = current.checkpoint_evaluation(model, train, data, 1)
            result["checkpoint_sha256"] = base.digest(path)
            result["legacy_config"] = config
            results[cell.name + "-" + name] = result
            base.dump(root / (cell.name + "-" + name + ".json"), result)
            print(cell.name, name, result["primary_mse"], flush=True)
    base.dump(root / "completion.json", {"checkpoint_evaluations": len(results), "cache_sha256": metadata["sha256"],
        "legacy_source_commit": json.loads((args.legacy / "provenance.json").read_text())["commit"]})
    assert len(results) == 12
    (root / "done").write_text("complete\n")


if __name__ == "__main__":
    main()
