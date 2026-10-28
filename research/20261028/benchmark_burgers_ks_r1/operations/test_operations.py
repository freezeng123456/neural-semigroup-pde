"""Scheduler-adapter failure tests; no login, Slurm mutation, data or training."""
import copy
import unittest
from unittest.mock import patch
import common
import scheduler_limits


def matrix():
    rows=[]
    for pde in common.PDES:
        for seed in range(3):
            for arm in common.ARMS:
                rows.append({'index':len(rows),'pde':pde,'arm':arm,'seed':seed,'run_name':f'{pde}-{arm}-seed{seed}'})
    return {'tasks':rows,'epochs':500,'updates_per_epoch':100,'batch_size':20,'max_concurrency':2}


class OperationTests(unittest.TestCase):
    def test_fixed_nine_plus_nine_partition(self):
        with patch.object(common,'load',return_value=matrix()):
            first,second=common.wave_tasks(1),common.wave_tasks(2)
        self.assertEqual(len(first),9);self.assertEqual(len(second),9)
        self.assertEqual({row['pde'] for row in first},{'burgers'})
        self.assertEqual({row['pde'] for row in second},{'ks'})
        self.assertEqual({row['index'] for row in first}|{row['index'] for row in second},set(range(18)))

    def test_missing_or_duplicated_scientific_arm_rejected(self):
        value=matrix();value['tasks'][-1]=copy.deepcopy(value['tasks'][-2]);value['tasks'][-1]['index']=17
        with patch.object(common,'load',return_value=value),self.assertRaises(ValueError):
            common.tasks()

    def test_unknown_squeue_gpu_is_not_zero(self):
        self.assertIsNone(scheduler_limits.gpu_count('N/A'))
        self.assertIsNone(scheduler_limits.gpu_count('(null)'))
        self.assertEqual(scheduler_limits.gpu_count('gres/gpu:tesla_v100:1'),1)

    def test_exact_terminal_allocation_success(self):
        text=''.join(f'123_{i}|{800+i}|COMPLETED|0:0\n' for i in range(9))
        with patch.object(common,'query',return_value={'stdout':text,'returncode':0}):
            report=common.accounting('123',9,require_success=True)
        self.assertEqual(len(report['allocation_rows']),9)

    def test_missing_or_failed_terminal_allocation_blocks_next_wave(self):
        for bad in ('missing','failed'):
            text=''.join(f'123_{i}|{800+i}|'+('FAILED|1:0' if bad=='failed' and i==8 else 'COMPLETED|0:0')+'\n' for i in range(8 if bad=='missing' else 9))
            with patch.object(common,'query',return_value={'stdout':text,'returncode':0}),self.assertRaises(ValueError):
                common.accounting('123',9,require_success=True)

    def test_queue_scoped_to_authorized_job_families(self):
        text='123_0|RUNNING\n1234_0|RUNNING\n456|PENDING\n999|RUNNING\n'
        with patch.object(common,'query',return_value={'command':['squeue'],'stdout':text,'returncode':0}):
            report=common.live_queue(['123','456'])
        self.assertEqual(report['matched_rows'],['123_0|RUNNING','456|PENDING'])


if __name__=='__main__':
    unittest.main()
