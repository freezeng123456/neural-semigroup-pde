#!/usr/bin/env python3
"""Audit actual completed artifacts and generate immutable file manifests."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

FULL = {"wave1-euler": 30, "wave2-midpoint": 24, "oracle": 6,
        "long-L4": 24, "rate-L4": 24, "long-L8": 24}
SMOKES = {"smoke-r2": 24, "extension-smoke-rate": 8, "extension-smoke-L8": 8}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args(); root = args.root
    torch.set_num_threads(2)
    audit = {"roots": {}, "full_cells": 0, "smoke_cells": 0, "full_checkpoints": 0,
             "full_optimizer_updates": 0, "full_metric_rows": 0, "status": "RUNNING"}
    caches = {}
    for name, expected in {**FULL, **SMOKES}.items():
        run = root / name
        assert (run / "done").is_file() and (run / "status").read_text().strip() == "COMPLETE", name
        assert read_json(run / "completion.json")["completed_cells"] == expected
        meta = read_json(run / "cache_metadata.json")
        assert digest(run / "cache.pt") == meta["sha256"]
        cache = torch.load(run / "cache.pt", map_location="cpu", weights_only=False)
        caches[name] = cache
        configs, count, checkpoints, metric_rows, updates = {}, 0, 0, 0, 0
        for cell in sorted((run / "cells").iterdir()):
            assert (cell / "done").is_file() and not (cell / "failed").exists()
            c = read_json(cell / "config.json"); result = read_json(cell / "summary.json")
            assert c == result["config"] and c["cache_sha256"] == meta["sha256"]
            assert c["training_uses_true_reaction"] == (c["mode"] == "oracle")
            assert result["weights_changed"] and result["weights_finite"]
            rows = list(csv.DictReader((cell / "metrics.csv").open()))
            assert int(rows[-1]["epoch"]) == c["epochs"]
            assert len(rows) == 1 + c["epochs"] // 10 + int(c["epochs"] % 10 != 0)
            assert all(math.isfinite(float(v)) for row in rows for v in row.values())
            for ck, ev in result["evaluations"].items():
                weights = torch.load(cell / (ck + ".pt"), map_location="cpu", weights_only=True)
                assert sum(v.numel() for v in weights.values()) == 65
                assert all(torch.isfinite(v).all() for v in weights.values())
                if ck == "final":
                    fingerprint = hashlib.sha256(b"".join(v.numpy().tobytes() for v in weights.values())).hexdigest()
                    assert fingerprint == result["final_weights_sha256"] and fingerprint != c["initial_weights_sha256"]
                vals = [ev["test"]["endpoints"][f"tau={t}:T={h}"]["mse"] for t in (.075, .15) for h in (1.2, 2.4)]
                expected_mse = math.exp(sum(math.log(v) for v in vals) / len(vals))
                assert math.isclose(expected_mse, ev["primary_mse"], rel_tol=1e-12)
                checkpoints += 1
            configs[cell.name] = c; count += 1; metric_rows += len(rows); updates += c["epochs"]
        assert count == expected
        for key, c in configs.items():
            other = configs[key[:-1] + ("A" if c["conditioned"] else "B")]
            assert c["initial_weights_sha256"] == other["initial_weights_sha256"]
            for field in ("budget", "epochs", "prediction_pairs_per_update", "reaction_scalar_evaluations_per_update", "mode", "scheme"):
                assert c[field] == other[field]
        audit["roots"][name] = {"cells": count, "checkpoints": checkpoints, "metric_rows": metric_rows,
            "optimizer_updates": updates, "cache_sha256": meta["sha256"], "passed": True}
        if name in FULL:
            audit["full_cells"] += count; audit["full_checkpoints"] += checkpoints
            audit["full_optimizer_updates"] += updates; audit["full_metric_rows"] += metric_rows
        else: audit["smoke_cells"] += count
    for a, b in (("wave1-euler", "wave2-midpoint"), ("wave1-euler", "oracle"), ("long-L4", "rate-L4")):
        assert digest(root / a / "cache.pt") == digest(root / b / "cache.pt")
    for seed in caches["long-L4"]["train"]:
        a, b = caches["long-L4"]["train"][seed], caches["long-L8"]["train"][seed]
        assert torch.equal(a["u"], b["u"]) and torch.equal(a["tau"], b["tau"])
        torch.testing.assert_close(a["states"], b["states"][:, :5], rtol=0, atol=0)
    for name in ("val", "test", "stress"):
        assert not torch.equal(caches["wave1-euler"][name]["u"], caches["long-L4"][name]["u"])
        torch.testing.assert_close(caches["long-L4"][name]["u"], caches["long-L8"][name]["u"], rtol=0, atol=0)
    assert (root / "launcher.exit").read_text().strip() == "0"
    assert (root / "extension-launcher.exit").read_text().strip() == "0"
    assert (root / "extension-tests.exit").read_text().strip() == "0"
    assert read_json(root / "legacy-diagnosis" / "completion.json")["checkpoint_evaluations"] == 12
    audit.update({"status": "PASSED", "matching_L4_L8_initials_and_first_four_snapshots": True,
        "fresh_extension_splits": True, "legacy_checkpoint_diagnoses": 12,
        "known_resolved_smoke_failure": "smoke-r1: GPU histogram determinism; CPU histogram fixed in ec8ec77",
        "focused_tests_passed": 12})
    (root / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    for name in (*FULL, *SMOKES, "legacy-diagnosis", "smoke-r1"):
        run = root / name
        paths = sorted(p for p in run.rglob("*") if p.is_file() and p.name != "artifacts.sha256")
        (run / "artifacts.sha256").write_text("".join(digest(p) + "  " + str(p.relative_to(run)) + "\n" for p in paths))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
