#!/usr/bin/env python3
"""Capture real scheduler state and bounded scalar logs, without launching work."""
import argparse
import hashlib
import json
from common import ROOT, live_queue, load, now, query, require_remote, tasks, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(); require_remote()
    waves = []
    for wave in (1, 2):
        path = ROOT / 'operations' / f'full_wave{wave}_submission_receipt.json'
        if path.exists():
            receipt = load(path)
            if receipt.get('status') != 'SUBMITTED' or not str(receipt.get('job_id')).isdigit():
                raise ValueError('A stage has an uncertain submission; reconcile actual Slurm before proceeding')
            waves.append({'wave':wave,'job':str(receipt['job_id'])})
    if not waves:
        raise ValueError('No actual submitted wave to monitor')
    monitored = [row['job'] for row in waves]
    recovery = None
    recovery_path = ROOT/'operations/recovery_submission_receipt.json'
    if recovery_path.exists():
        submitted = load(recovery_path)
        if submitted.get('status') != 'SUBMITTED' or not str(submitted.get('job_id')).isdigit():
            raise ValueError('Uncertain recovery submission requires actual Slurm reconciliation')
        recovery = str(submitted['job_id']); monitored.append(recovery)
    state = {'schema':'burgers_ks_progress_v1','captured_at_utc':now(),'waves':waves,
             'recovery_job':recovery,
             'scheduler': [query(['sacct','-nP','-j',job,'--format=JobID,JobIDRaw,State,ExitCode,Elapsed']) for job in monitored],
             'live_queue':live_queue(monitored),
             'global_failure_marker':(ROOT/'operations/FULL_FAILURE.json').exists(),'tasks':[]}
    for name in ('RECOVERY_EXECUTION_RECEIPT.json','RECOVERY_EXECUTION_FAILURE.json'):
        if (ROOT/'operations'/name).exists():
            state[name] = load(ROOT/'operations'/name)
    for task in tasks():
        directory = ROOT / 'runs' / task['run_name']; row = {**task,'directory_exists':directory.exists()}
        for leaf in ('FAILURE.json','RUN_RECEIPT.json','update000001.json'):
            if (directory/leaf).exists():
                row[leaf] = load(directory/leaf)
        path = directory/'epochs.jsonl'
        if path.exists():
            lines = []
            for text in path.read_text().splitlines():
                try:
                    lines.append(json.loads(text))
                except json.JSONDecodeError:
                    pass  # A concurrently appended final line can be incomplete.
            row.update(epoch_rows=len(lines),first_epoch=lines[0] if lines else None,latest_epoch=lines[-1] if lines else None)
        validations = sorted(directory.glob('validation_epoch*.json'))
        if validations:
            row['latest_validation'] = load(validations[-1])
        row['latest_checkpoint_exists'] = (directory/'latest_state.pt').exists()
        state['tasks'].append(row)
    name = 'progress-' + state['captured_at_utc'].replace(':','').replace('-','').replace('.','') + '.json'
    path = ROOT/'operations'/name; write(path,state)
    print(json.dumps({'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'global_failure_marker':state['global_failure_marker'],
                      'summaries':[{'run':row['run_name'],'updates':(row.get('latest_epoch') or {}).get('updates'),
                                    'loss':(row.get('latest_epoch') or {}).get('total'),'failure':'FAILURE.json' in row} for row in state['tasks']]},indent=2),flush=True)


if __name__ == '__main__':
    main()
