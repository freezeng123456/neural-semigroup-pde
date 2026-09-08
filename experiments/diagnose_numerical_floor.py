#!/usr/bin/env python3
"""Privileged exact-source solver diagnostics, never used by training."""
import argparse
import json
from pathlib import Path

import torch
import run_generator_coverage as current
from run_generator_coverage import base


class ExactSource(current.Flow):
    def reaction(self, u, conditioning):
        return base.truth_reaction(u)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    results = {}
    for split in ("wave1-euler", "long-L4"):
        cache = args.root / split / "cache.pt"
        expected_hash = json.loads((cache.parent / "cache_metadata.json").read_text())["sha256"]
        assert base.digest(cache) == expected_hash
        data = torch.load(cache, map_location="cpu", weights_only=False)
        models = {"pure_heat": base.OracleSplit(pure_heat=True), "known_cubic_only": base.OracleSplit(4, pure_cubic=True),
            "exact_source_euler_b4": ExactSource(False, 4, "euler"),
            "exact_source_midpoint_b4": ExactSource(False, 4, "midpoint"),
            "exact_source_midpoint_b16": ExactSource(False, 16, "midpoint")}
        results[split] = {"cache_sha256": expected_hash, "models": {}}
        for name, model in models.items():
            evaluations = {key: base.evaluate(model, data[key]) for key in ("test", "stress")}
            evaluations["primary_mse"] = base.gm([evaluations["test"]["endpoints"][f"tau={t}:T={h}"]["mse"] for t in (.075, .15) for h in (1.2, 2.4)])
            results[split]["models"][name] = evaluations
            print(split, name, evaluations["primary_mse"], flush=True)
    base.dump(args.output / "baselines.json", {"privileged_diagnostic": True, "device": "cpu",
        "note": "Exact-source solver error is diagnostic; it is not a mathematical lower bound on learned discrete-map error.",
        "results": results})
    (args.output / "done").write_text("complete\n")


if __name__ == "__main__":
    main()
