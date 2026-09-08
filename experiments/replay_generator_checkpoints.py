#!/usr/bin/env python3
"""Independently reload and replay every selected full-run checkpoint on CPU."""
import argparse
import json
from pathlib import Path
import time

import torch
import run_generator_coverage as current
from run_generator_coverage import base


def main():
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    args = p.parse_args(); root = args.root
    torch.set_num_threads(2)
    while not (root / "extension-launcher.exit").exists(): time.sleep(5)
    assert (root / "extension-launcher.exit").read_text().strip() == "0"
    rows = []
    for name in ("wave1-euler", "wave2-midpoint", "oracle", "long-L4", "rate-L4", "long-L8"):
        data = torch.load(root / name / "cache.pt", map_location="cpu", weights_only=False)
        for path in sorted((root / name / "cells").glob("*/summary.json")):
            r = json.loads(path.read_text()); c = r["config"]
            model = current.Flow(c["conditioned"], c["budget"], c["scheme"])
            checkpoint = path.parent / "best_rollout.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            actual = base.evaluate(model, data["test"])
            expected = r["evaluations"]["best_rollout"]["test"]
            differences = {}
            for key, value in actual["endpoints"].items():
                before, after = expected["endpoints"][key]["mse"], value["mse"]
                differences[key] = abs(after - before)
                # Cross-device FP32 replay, deliberately documented tolerance.
                assert abs(after - before) <= max(1e-8, .02 * before), (name, path.parent.name, key, before, after)
            rows.append({"root": name, "cell": path.parent.name, "checkpoint_sha256": base.digest(checkpoint),
                "max_absolute_endpoint_mse_difference": max(differences.values()), "passed": True})
        print("REPLAY_COMPLETE", name, len(rows), flush=True)
    assert len(rows) == 132
    base.dump(root / "CHECKPOINT_REPLAY.json", {"status": "PASSED", "selected_checkpoints_replayed": len(rows),
        "device": "cpu", "dtype": "float32", "endpoints_per_checkpoint": 9,
        "tolerance": "absolute MSE difference <= max(1e-8, 0.02 * original GPU MSE)", "rows": rows})


if __name__ == "__main__": main()
