"""Optional native-graph checks requiring the recorded paper dependencies."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from public_common import ROOT, read, check_environment
import public_graph as graph


class GraphProtocol(unittest.TestCase):
    def test_independent_atom_bond_and_reverse_encoding(self):
        check_environment(graph=True)
        schema=read(ROOT/'configs/graph_schema.json')
        for smiles in ['CCO','c1ccccc1','[Na+].[O-]C=O']:
            graph.compare_graph(graph.graph(smiles),graph.independent_graph(smiles,schema))

    def test_synthetic_fixed_epoch_fit_reload_and_prediction(self):
        import torch
        hp=dict(hidden_size=300,depth=3,dropout=0.,learning_rate=.001,batch_size=2,max_epochs=1,seed=1)
        graph.setup(1,'cpu')
        net=graph.model(hp).to('cpu')
        dataset=graph.GraphRows([graph.graph('CCO'),graph.graph('CC')],['a','b'],[0.,1.])
        blind=graph.GraphRows(dataset.graphs,dataset.row_ids)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            history=graph.fit_graph_fixed(net,dataset,hp,'cpu',out=folder)
            self.assertEqual(history['epochs_run'],1)
            prediction=graph.predict(net,blind,2,'cpu')
            reloaded=graph.model(hp)
            reloaded.load_state_dict(torch.load(folder/'final_model.pt',map_location='cpu',weights_only=True))
            np.testing.assert_array_equal(prediction,graph.predict(reloaded,blind,2,'cpu'))
            self.assertTrue(np.isfinite(prediction).all())


if __name__=='__main__':unittest.main()
