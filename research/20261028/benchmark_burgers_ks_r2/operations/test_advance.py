"""Coordinator ordering and failure gates; no scheduler, data or optimization."""
import unittest
from unittest.mock import patch
import advance
import submit


class AdvanceTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(advance.os.environ,{'SLURM_JOB_ID':'150'})
        environment.start();self.addCleanup(environment.stop)

    def test_failed_wave1_does_not_call_any_submission(self):
        calls=[]
        with patch.object(advance,'audit_completed_wave',side_effect=ValueError('incomplete')):
            with self.assertRaises(ValueError):
                advance.follow_stages({'preflight_job':'100'},calls.append)
        self.assertEqual(calls,[])

    def test_audit_before_each_mutation_and_record_real_job_ids(self):
        calls=[]
        with patch.object(advance,'audit_completed_wave',return_value={'passed':True}), \
             patch.object(advance,'submitted_job',side_effect=['201','202']), \
             patch.object(advance,'load',return_value={'command':['--dependency=afterok:150']}):
            gate,wave,recovery=advance.follow_stages({'preflight_job':'100'},calls.append)
        self.assertEqual((wave,recovery),('201','202')); self.assertTrue(gate['passed'])
        self.assertEqual(len(calls),4)
        self.assertEqual(calls[0],calls[1]+['--audit-only'])
        self.assertEqual(calls[2],calls[3]+['--audit-only'])
        self.assertIn('--preflight-job',calls[0]); self.assertEqual(calls[1][-3],'100')
        self.assertEqual(calls[1][-2:],['--advance-job','150'])

    def test_failed_wave2_audit_prevents_mutation_and_recovery(self):
        calls=[]
        def fail(command):
            calls.append(command); raise RuntimeError('audit failed')
        with patch.object(advance,'audit_completed_wave',return_value={}),self.assertRaises(RuntimeError):
            advance.follow_stages({'preflight_job':'100'},fail)
        self.assertEqual(len(calls),1); self.assertEqual(calls[0][-1],'--audit-only')

    def test_uncertain_wave2_receipt_prevents_recovery(self):
        calls=[]
        with patch.object(advance,'audit_completed_wave',return_value={}), \
             patch.object(advance,'submitted_job',side_effect=ValueError('unknown')), \
             self.assertRaises(ValueError):
            advance.follow_stages({'preflight_job':'100'},calls.append)
        self.assertEqual(len(calls),2)

    def test_recovery_audit_failure_cannot_repeat_wave2(self):
        calls=[]
        def fail_third(command):
            calls.append(command)
            if len(calls)==3:raise RuntimeError('recovery audit failed')
        with patch.object(advance,'audit_completed_wave',return_value={}), \
             patch.object(advance,'submitted_job',return_value='201'), \
             patch.object(advance,'load',return_value={'command':['--dependency=afterok:150']}), \
             self.assertRaises(RuntimeError):
            advance.follow_stages({'preflight_job':'100'},fail_third)
        self.assertEqual(len(calls),3)

    def test_non_numeric_or_uncertain_receipt_rejected(self):
        for row in ({'schema':'s','status':'UNKNOWN_OR_FAILED_RECONCILE_SLURM','job_id':'123'},
                    {'schema':'s','status':'SUBMITTED','job_id':None},
                    {'schema':'wrong','status':'SUBMITTED','job_id':'123'}):
            with patch.object(advance,'load',return_value=row),self.assertRaises(ValueError):
                advance.submitted_job('unused','s')

    def test_default_wave_submission_has_no_coordinator_dependency(self):
        self.assertIsNone(submit.advance_gate(None,'full',2))

    def test_coordinator_dependency_is_restricted_to_current_wave2_allocation(self):
        for job,mode,wave in [('150','preflight',None),('150','full',1),('151','full',2),('bad','full',2)]:
            with self.assertRaises(ValueError):submit.advance_gate(job,mode,wave)

    def test_dependency_binds_exact_durable_execution_and_payload(self):
        identity={'core_manifest_sha256':'abc'}
        receipt={'schema':'burgers_ks_advance_submission_v1','status':'SUBMITTED','job_id':'150',
                 'payload':{'source':'bytes'},'identity':identity}
        execution={'identity':identity,'slurm_job_id':'150','optimizer_updates':0}
        with patch.object(submit,'load',side_effect=[receipt,execution]), \
             patch.object(submit,'operations_payload',return_value={'source':'bytes'}), \
             patch.object(submit,'sha',return_value='abc'):
            result=submit.advance_gate('150','full',2)
        self.assertEqual(result['dependency'],'afterok:150')


if __name__=='__main__':
    unittest.main()
