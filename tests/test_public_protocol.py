"""Independent checks of scientific invariants, failure handling and IO boundaries."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import public_common as c
from c05_reproduction import calculate_smiles
from public_models import validation_jobs


class PublicProtocol(unittest.TestCase):
    def test_training_only_imputation_and_filtering(self):
        train = np.array([[1,np.nan,7,np.nan],[3,2,7,np.nan],[np.nan,4,7,np.nan]],dtype=float)
        processor, available, selected = c.fit_processor(train)
        self.assertEqual(available.tolist(),[0,1,2])
        self.assertEqual(selected.tolist(),[0,1])
        np.testing.assert_array_equal(processor.named_steps['imputer'].statistics_,[2,3,7])
        before=processor.named_steps['imputer'].statistics_.copy()
        heldout=np.array([[1000,np.nan,99,8],[-1000,1000,5,2]],dtype=float)
        transformed=processor[:-1].transform(heldout[:,available])
        np.testing.assert_array_equal(transformed,[[1000,3],[-1000,1000]])
        np.testing.assert_array_equal(processor.named_steps['imputer'].statistics_,before)

    def test_metrics_and_constant_rank(self):
        measured=c.metrics([0,1,2],[0,2,1])
        self.assertAlmostEqual(measured['MAE'],2/3)
        self.assertAlmostEqual(measured['RMSE'],np.sqrt(2/3))
        self.assertAlmostEqual(measured['R2'],0)
        self.assertAlmostEqual(measured['Spearman'],.5)
        self.assertIsNone(c.metrics([0,1,2],[1,1,1])['Spearman'])

    def test_large_finite_negative_results_are_retained(self):
        measured=c.metrics([0,1,2],[1e108,1e108,1e108])
        self.assertGreater(measured['MAE'],1e107)
        self.assertLess(measured['R2'],-1e210)
        self.assertTrue(np.isfinite(measured['R2']))
        self.assertTrue(np.isfinite(c.stable_sd([0,-1e220,0,0,0])))
        self.assertAlmostEqual(float(c.stable_sd([0,-1e220,0,0,0])) / 1e220, 1/np.sqrt(5))

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaises(ValueError):c.metrics([0,1],[0,float('nan')])

    def test_structure_failures_are_explicit(self):
        for ids,smiles in [(['a'],['']),(['a'],['not_a_smiles']),(['a','a'],['CC','CCO'])]:
            with self.assertRaises(ValueError):calculate_smiles(ids,smiles)

    def test_morgan_binary_and_descriptor_order(self):
        bits,desc=calculate_smiles(['ethanol'],['CCO'])
        self.assertEqual(bits.shape,(1,2048))
        self.assertEqual(bits.dtype,np.uint8)
        self.assertTrue(np.isin(bits,[0,1]).all())
        self.assertEqual(desc.shape,(1,210))
        names=c.read(c.ROOT/'manifests/c03/feature_schema.json')['rdkit2d']['feature_names']
        self.assertAlmostEqual(desc[0,names.index('MolWt')],46.069,places=3)

    def test_full_validation_budget_and_fixed_seeds(self):
        jobs=validation_jobs('paper')
        self.assertEqual(len(jobs),465)
        self.assertEqual(len({j['job_id'] for j in jobs}),465)
        self.assertEqual(sum(j['algorithm']=='dmpnn' for j in jobs),60)
        primary=validation_jobs('primary')
        self.assertEqual(len(primary),60)
        for job in primary:
            self.assertFalse(job['parameters']['use_best_model'])
            self.assertEqual(job['parameters']['random_seed'],job['split_seed'])

    def test_run_traversal_rejected(self):
        for name in ['../outside','/tmp/outside','.','..']:
            with self.assertRaises(ValueError):c.safe_run(name)

    def test_development_denies_held_out_data(self):
        with tempfile.TemporaryDirectory() as directory:
            test=Path(directory)/'test.csv';test.write_text('Y\n1\n')
            source="import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import public_common as c; c.DATA=Path(sys.argv[2]); c.development_guard(); (c.DATA/'test.csv').read_text()"
            result=subprocess.run([sys.executable,'-B','-c',source,str(c.ROOT/'code'),directory],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Held-out data are forbidden',result.stderr)


if __name__=='__main__':unittest.main()
