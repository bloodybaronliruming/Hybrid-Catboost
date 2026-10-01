"""Native graph and D-MPNN arithmetic extracted from the accepted v1 source."""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from chemprop.data import MoleculeDatapoint, MolGraph
from chemprop.data.datasets import Datum
from chemprop.data.collate import collate_batch
from chemprop.featurizers import MultiHotAtomFeaturizer, MultiHotBondFeaturizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from public_common import require, sha, dump

def graph(smiles):
    require(isinstance(smiles,str) and bool(smiles.strip()),'Empty/non-string SMILES')
    point=MoleculeDatapoint.from_smi(smiles,y=None,keep_h=False,add_h=False,ignore_stereo=False,reorder_atoms=False)
    require(point.mol is not None and point.mol.GetNumAtoms()>0,'Invalid/empty graph')
    f=SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2(),bond_featurizer=MultiHotBondFeaturizer(),extra_atom_fdim=0,extra_bond_fdim=0)
    g=f(point.mol)
    return MolGraph(np.asarray(g.V,dtype=np.float32),np.asarray(g.E,dtype=np.float32),np.asarray(g.edge_index,dtype=np.int64),np.asarray(g.rev_edge_index,dtype=np.int64))

def independent_graph(smiles,record):
    """Manual RDKit atom/bond encoding; does not call the native featurizers."""
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    require(mol is not None and mol.GetNumAtoms()>0,'Independent invalid/empty molecule')
    properties=['atomic_nums','degrees','formal_charges','chiral_tags','num_Hs','hybridizations'];vertices=[]
    for atom in mol.GetAtoms():
        values=[atom.GetAtomicNum(),atom.GetTotalDegree(),atom.GetFormalCharge(),int(atom.GetChiralTag()),int(atom.GetTotalNumHs()),str(atom.GetHybridization())]
        encoded=[]
        for key,value in zip(properties,values):
            choices=record['atom_categories'][key];bits=[0.0]*(len(choices)+1);bits[choices.index(value) if value in choices else len(choices)]=1.;encoded+=bits
        encoded += [float(atom.GetIsAromatic()),0.01*atom.GetMass()];vertices.append(encoded)
    bonds=[];edges=[]
    for b in mol.GetBonds():
        bits=[0.0]*14;t=str(b.GetBondType())
        if t in record['bond_types']:bits[1+record['bond_types'].index(t)]=1.
        bits[5]=float(b.GetIsConjugated());bits[6]=float(b.IsInRing());st=int(b.GetStereo())
        bits[7+(record['bond_stereo'].index(st) if st in record['bond_stereo'] else len(record['bond_stereo']))]=1.
        bonds.extend([bits,bits]);u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx();edges.extend([(u,v),(v,u)])
    edge=np.asarray(edges,dtype=np.int64).reshape(-1,2).T
    reverse=np.arange(len(edges),dtype=np.int64)^1
    return MolGraph(np.asarray(vertices,dtype=np.float32),np.asarray(bonds,dtype=np.float32).reshape(-1,14),edge,reverse)

def compare_graph(actual,expected):
    require(actual.V.shape[1]==72 and actual.E.shape[1]==14,'Graph width differs')
    for key in ['V','E','edge_index','rev_edge_index']:
        a=getattr(actual,key);b=getattr(expected,key);require(np.array_equal(a,b),'Independent graph differs: '+key);require(np.isfinite(a).all(),'Nonfinite graph')
    rev=actual.rev_edge_index;n=len(rev)
    require(np.array_equal(rev[rev],np.arange(n)) and np.array_equal(actual.edge_index[:,rev],actual.edge_index[::-1]),'Reverse-edge mapping invalid')

class GraphRows(Dataset):
    def __init__(self,graphs,row_ids,labels=None):
        require(len(graphs)==len(row_ids)>0 and len(set(row_ids))==len(row_ids),'Duplicate/missing graph rows')
        require(labels is None or len(labels)==len(row_ids),'Label mapping differs')
        self.graphs=graphs;self.row_ids=list(row_ids);self.labels=None if labels is None else np.asarray(labels,dtype=np.float32)
    def __len__(self):return len(self.row_ids)
    def __getitem__(self,index):
        target=None if self.labels is None else np.asarray([self.labels[index]],dtype=np.float32)
        return self.row_ids[index],Datum(self.graphs[index],None,None,target,1.0,None,None)

def collate(items):
    names,datums=zip(*items);return list(names),collate_batch(datums)

def loader(dataset,batch_size,seed=1,shuffle=False):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,drop_last=False,pin_memory=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed))

def model(hp):
    message=BondMessagePassing(d_v=72,d_e=14,d_h=hp['hidden_size'],bias=False,depth=hp['depth'],dropout=hp['dropout'],activation='relu',undirected=False,d_vd=None,V_d_transform=None,graph_transform=None)
    head=RegressionFFN(n_tasks=1,input_dim=hp['hidden_size'],hidden_dim=hp['hidden_size'],n_layers=2,dropout=hp['dropout'],activation='relu',criterion=None,task_weights=None,threshold=None,output_transform=None)
    net=MPNN(message,MeanAggregation(dim=0),head,batch_norm=False,metrics=None,warmup_epochs=0,init_lr=hp['learning_rate'],max_lr=hp['learning_rate'],final_lr=hp['learning_rate'],X_d_transform=None)
    dims=[(m.in_features,m.out_features) for m in head.ffn.modules() if isinstance(m,torch.nn.Linear)];h=hp['hidden_size']
    require(dims==[(h,h),(h,h),(h,1)],'Native FFN architecture differs')
    return net

def setup(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    require(device in ['cpu','cuda:0'],'Unapproved device')
    if device=='cuda:0':require(torch.cuda.is_available(),'GPU unavailable; CPU fallback prohibited');torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def predict(net,dataset,batch_size,device):
    net.eval();values=[];ids=[]
    with torch.no_grad():
        for names,batch in loader(dataset,batch_size):
            require(batch.Y is None and batch.V_d is None and batch.X_d is None,'Prediction batch contains labels/extra fields')
            batch.bmg.to(device);p=net(batch.bmg,None,None).reshape(-1)
            require(torch.isfinite(p).all(),'Nonfinite prediction');values.extend(p.cpu().numpy().astype(float));ids.extend(names)
    require(ids==dataset.row_ids,'Prediction row order differs')
    return np.asarray(values,dtype=float)

def fit(net,train,valid,y_valid,hp,device,out,progress=None):
    """Native MPNN + direct Adam/MSE. No Lightning Trainer or implicit scheduler."""
    optimizer=torch.optim.Adam(net.parameters(),lr=hp['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    batches=loader(train,hp['batch_size'],seed=hp['seed'],shuffle=True)
    history=[];best=float('inf');stale=0;best_epoch=None;checkpoint=out/'best_model.pt'
    for epoch in range(1,hp['max_epochs']+1):
        net.train();seen=[];loss_sum=0.;gradient_checked=False
        for names,batch in batches:
            require(batch.Y is not None and batch.V_d is None and batch.X_d is None,'Invalid training batch')
            batch.bmg.to(device);target=batch.Y.to(device).reshape(-1);optimizer.zero_grad(set_to_none=True)
            p=net(batch.bmg,None,None).reshape(-1);loss=torch.nn.functional.mse_loss(p,target)
            require(torch.isfinite(loss),'Nonfinite training loss');loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Nonfinite gradient')
            gradient_checked=True;optimizer.step();loss_sum+=float(loss.item())*len(names);seen+=names
        require(len(seen)==len(train) and set(seen)==set(train.row_ids),'Epoch members differ')
        prediction=predict(net,valid,hp['batch_size'],device);score=float(np.mean(np.abs(prediction-y_valid)))
        improved=score<best
        if improved:
            best=score;stale=0;best_epoch=epoch
            torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},checkpoint)
        else:stale+=1
        history.append(dict(epoch=epoch,train_mse=loss_sum/len(train),valid_mae=score,improved=improved,stale_epochs=stale,
                            train_row_ids_in_order=seen,gradient_finite=gradient_checked,checkpoint_sha256=sha(checkpoint)))
        if progress:progress(epoch,score)
        if stale>=hp['patience']:break
    net.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));net.to(device)
    prediction=predict(net,valid,hp['batch_size'],device)
    require(np.isclose(np.mean(np.abs(prediction-y_valid)),best,rtol=1e-6,atol=1e-6),'Best checkpoint prediction differs')
    return prediction,dict(history=history,best_epoch=best_epoch,best_valid_mae=best,epochs_run=len(history),stop_reason='patience' if stale>=hp['patience'] else 'max_epochs',fit_row_ids=train.row_ids,
                           validation_row_ids=valid.row_ids,optimizer='torch.optim.Adam',loss='MSE',scheduler=None,label_transform='identity')

def fit_graph_fixed(net, dataset, hp, device, out=None, progress=None):
    """The B6 fixed-epoch final loop has no validation dataset or scorer argument."""
    import torch
    require(isinstance(hp['max_epochs'], int) and hp['max_epochs'] > 0 and dataset.labels is not None, 'Invalid fixed-epoch training budget/dataset')
    optimizer = torch.optim.Adam(net.parameters(), lr=hp['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
    batches = loader(dataset, hp['batch_size'], seed=hp['seed'], shuffle=True)
    history = []
    for epoch in range(1, hp['max_epochs'] + 1):
        net.train(); seen = []; loss_sum = 0.
        for names, batch in batches:
            require(batch.Y is not None and batch.X_d is None and batch.V_d is None, 'Invalid final training batch')
            batch.bmg.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = net(batch.bmg, None, None).reshape(-1)
            loss = torch.nn.functional.mse_loss(prediction, batch.Y.to(device).reshape(-1))
            require(torch.isfinite(loss), 'Nonfinite final training loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()), 'Nonfinite final gradients')
            optimizer.step(); seen += names; loss_sum += float(loss.item()) * len(names)
        require(len(seen) == len(dataset) and set(seen) == set(dataset.row_ids), 'Final epoch membership differs')
        event = dict(epoch=epoch, train_mse=loss_sum / len(dataset), train_row_ids_in_order=seen, gradient_finite=True)
        history.append(event)
        if out:
            dump(Path(out) / f'epoch_{epoch:03d}.json', event)
        if progress:
            progress(epoch, event['train_mse'])
    require(all(torch.isfinite(p).all() for p in net.state_dict().values()), 'Nonfinite final weights')
    if out:
        torch.save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, Path(out) / 'final_model.pt')
    return dict(epochs_run=len(history), fixed_epochs=hp['max_epochs'], fit_row_ids=dataset.row_ids,
                optimizer='Adam', loss='MSE', scheduler=None, early_stopping=False, validation_scoring=False,
                checkpoint='last fixed epoch', history=history)
