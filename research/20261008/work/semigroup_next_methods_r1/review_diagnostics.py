#!/usr/bin/env python3
"""Independently recompute frozen diagnostics; exclude initial roundoff gain."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def stat(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.quantile(a, .9)), 'max': float(a.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--stage-b', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True, exist_ok=False)
    protected = {a.data: sha(a.data)}
    manifest = json.loads((a.input/'artifact_manifest.json').read_text())
    for name, info in manifest.items():
        f = a.input/name
        assert f.stat().st_size == info['bytes'] and sha(f) == info['sha256']
        protected[f] = info['sha256']
    report = json.loads((a.input/'DIAGNOSTIC_REPORT.json').read_text())
    with np.load(a.data, allow_pickle=False) as z:
        truth = z['validation_truth'][:, :, :, 0, :].transpose(0, 3, 1, 2).astype(np.float64)
    k = np.fft.fftfreq(64)*64
    kmax = np.maximum(np.abs(k[:, None]), np.abs(k[None, :]))
    mask = (kmax == 0, (kmax > 0)&(kmax < 4), (kmax >= 4)&(kmax < 8),
            (kmax >= 8)&(kmax < 12), kmax >= 12)
    bands = ('zero', '0<kmax<4', '4<=kmax<8', '8<=kmax<12', 'kmax>=12')
    checks = 0
    corrections = []
    with np.load(a.input/'diagnostic_output_fields.npz', allow_pickle=False) as fields, np.load(a.input/'diagnostic_metrics.npz', allow_pickle=False) as old:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        colors = ['#366ac5', '#db7b26', '#40916c']
        for row in report['spectra']:
            arm, seed = row['arm'], row['seed']
            name = f'{arm}-seed{seed}-r1'
            run = a.stage_b/'training-recovery'/name/'20261004-midpoint-stage-b-r1'/'runs'/name
            path = run/'validation_final_prediction.npy'
            digest = sha(path)
            expected = json.loads((run/'artifact_manifest.json').read_text())['validation_final_prediction.npy']['sha256']
            assert digest == expected
            protected[path] = digest
            pred = np.load(path, allow_pickle=False)[:, :, :, 0, :].transpose(0, 3, 1, 2).astype(np.float64)
            err = pred[:, 1:]-truth[:, 1:]
            # Independent default FFT, with explicit Parseval normalization.
            power = np.abs(np.fft.fft2(err, axes=(-2, -1)))**2 / 4096
            energy = np.stack([power[..., m].sum(-1) for m in mask], -1)
            assert np.allclose(energy.sum(-1), (err*err).sum((-2, -1)), rtol=2e-12, atol=1e-20)
            assert np.allclose(energy, old[name+'_error_band_energy'], rtol=2e-12, atol=1e-20)
            l2 = np.sqrt((err*err).sum((-2, -1))/(truth[:, 1:]**2).sum((-2, -1)))
            assert np.allclose(l2, old[name+'_rollout_l2'], rtol=2e-12, atol=1e-14)
            assert np.isclose(l2.mean(), row['rollout_positive_snapshot_relative_l2']['mean'], rtol=2e-12)
            checks += 4
            if arm != 'gdsg500':
                continue
            tf = fields[name+'_teacher_forced_prediction'].astype(np.float64)
            nxt = fields[name+'_from_saved_previous_prediction'].astype(np.float64)
            d = tf-truth[:, 1:]
            q = nxt-tf
            e = nxt-truth[:, 1:]
            assert np.array_equal(e, q+d)
            ep = pred[:, :-1]-truth[:, :-1]
            pn = np.sqrt((ep*ep).sum((-2, -1)))
            qn = np.sqrt((q*q).sum((-2, -1)))
            dn = np.sqrt((d*d).sum((-2, -1)))
            gain = np.full(pn.shape, np.nan)
            valid = pn > 0
            valid[:, 0] = False
            np.divide(qn, pn, out=gain, where=valid)
            cosine = np.full(pn.shape, np.nan)
            np.divide((q*d).sum((-2, -1)), qn*dn, out=cosine, where=valid&(qn*dn>0))
            tfl2 = np.sqrt((d*d).sum((-2, -1))/(truth[:, 1:]**2).sum((-2, -1)))
            assert np.allclose(tfl2, old[name+'_teacher_forced_l2'], rtol=2e-12, atol=1e-14)
            assert np.allclose(gain[:, 1:], old[name+'_directional_gain'][:, 1:], rtol=2e-12, atol=1e-14)
            cpu_diff = np.sqrt(((nxt-pred[:, 1:])**2).sum((-2, -1))/(pred[:, 1:]**2).sum((-2, -1)))
            assert cpu_diff.max() < 2e-5
            init_rel = np.sqrt((ep[:, 0]**2).sum((-2, -1))/(truth[:, 0]**2).sum((-2, -1)))
            initial_report = report['local_propagation'][seed]
            assert initial_report['seed'] == seed
            assert np.isclose(tfl2.mean(), initial_report['teacher_forced_relative_l2']['mean'], rtol=2e-12)
            corrections.append({'seed': seed, 'teacher_forced_l2': stat(tfl2), 'rollout_l2': stat(l2),
                                'rollout_to_teacher_forced_mean_ratio': float(l2.mean()/tfl2.mean()),
                                'directional_gain_excluding_initial': stat(gain),
                                'directional_gain_fraction_above_one': float(np.mean(gain[np.isfinite(gain)] > 1)),
                                'alignment_excluding_initial': stat(cosine),
                                'initial_saved_state_max_relative_roundoff': float(init_rel.max()),
                                'cpu_gpu_next_state_max_relative_l2': float(cpu_diff.max()),
                                'error_energy_share_by_band': row['error_energy_share_by_band']})
            checks += 7
            axs[0].plot(np.arange(1, 51), l2.mean(0)*100, color=colors[seed], label=f'Rollout, seed {seed}')
            axs[0].plot(np.arange(1, 51), tfl2.mean(0)*100, color=colors[seed], linestyle='--', label=f'True input, seed {seed}')
            shares = energy.sum((0, 1))/energy.sum()
            axs[1].bar(np.arange(5)+(seed-1)*.24, shares*100, width=.23, color=colors[seed], label=f'seed {seed}')
            axs[2].plot(np.arange(2, 51), gain[:, 1:].mean(0), color=colors[seed], label=f'seed {seed}')
        axs[0].set(xlabel='Snapshot index', ylabel='Mean physical relative L2 (%)', title='Local and rollout error')
        axs[0].legend(fontsize=7, ncol=2)
        axs[1].set(xticks=np.arange(5), xticklabels=['Mean', '0<k<4', '4<=k<8', '8<=k<12', 'k>=12'], ylabel='Share of squared rollout error (%)', title='Error by spatial frequency')
        axs[1].tick_params(axis='x', labelsize=8)
        axs[1].legend(fontsize=8)
        axs[2].axhline(1, color='#666666', linestyle='--', linewidth=1)
        axs[2].set(xlabel='Snapshot index (initial excluded)', ylabel='Mean directional gain', title='Response along realized rollout error')
        axs[2].legend(fontsize=8)
        for ax in axs:
            ax.grid(alpha=.18)
        fig.suptitle('Frozen GDSG-500: 3 seeds, 20 development trajectories; no training updates', fontsize=11)
        fig.savefig(a.output/'diagnostic_overview.png', dpi=180)
        fig.savefig(a.output/'diagnostic_overview.pdf')
        plt.close(fig)
    for f, digest in protected.items():
        assert sha(f) == digest
    result = {'status': 'INDEPENDENT_DIAGNOSTIC_REVIEW_PASS_WITH_INITIAL_GAIN_CORRECTION',
              'checks': checks, 'optimizer_updates': 0, 'additional_network_forwards': 0,
              'protected_inputs_unchanged': len(protected),
              'r1_report_sha256': sha(a.input/'DIAGNOSTIC_REPORT.json'), 'review_script_sha256': sha(__file__),
              'correction': 'The saved GPU initial state differs from raw truth by float32 normalization roundoff. R1 wrongly included its tiny input-error gain. Exclude all 20 first transitions for each seed from gain and alignment summaries; accuracy and spectral summaries are unchanged.',
              'r1_gain_and_alignment_summaries_superseded': True,
              'no_epsilon_or_sample_selection': True, 'corrected_gdsg500': corrections}
    (a.output/'INDEPENDENT_REVIEW.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    files = {f.name: {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in a.output.iterdir() if f.is_file()}
    (a.output/'artifact_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
