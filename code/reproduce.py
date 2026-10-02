"""Public, staged raw-data-to-result reproduction of the frozen solubility study."""
import os
import sys
import json
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
runtime = json.loads((Path(__file__).resolve().parents[1] / 'configs/numerical_runtime.json').read_text())
feature_stage = len(sys.argv) > 1 and sys.argv[1] in ('features', 'predict-smiles')
thread_count = runtime['feature_blas_threads'] if feature_stage else runtime['training_blas_threads']
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(thread_count)
import argparse
from public_common import (ROOT, SEEDS, require, recipe, safe_run, read, dump,
    check_environment, development_guard, split_frame, view, matrix_sha, write_csv)


def check(scope):
    from public_models import binding, validation_jobs
    from catboost.utils import get_gpu_device_count
    bound = binding(scope)
    matrices = []
    for spec in recipe()['historical_final']:
        if spec['experiment'] != 'S32_CATBOOST':
            continue
        train = split_frame(spec['split_seed'], 'train')
        x, _ = view(spec, 'train_val', train.source_row_index.to_numpy(dtype=int))
        require(matrix_sha(x) == spec['matrix_sha256'], 'Historical primary matrix differs')
        matrices.append(dict(seed=spec['split_seed'],rows=len(x),columns=x.shape[1],matrix_sha256=matrix_sha(x)))
    gpu = get_gpu_device_count()
    if scope == 'paper':
        import torch
        require(torch.cuda.is_available(), 'CUDA required for full-paper D-MPNN')
    print(json.dumps(dict(status='READY' if gpu > 0 else 'BLOCKED_NO_GPU',scope=scope,
        available_gpu_devices=gpu,validation_fits=len(validation_jobs(scope)),final_fits=50 if scope=='paper' else 5,
        frozen_training_views=matrices,environment=bound['environment'])))


def predict_smiles(run_name, source, target):
    import numpy as np
    import pandas as pd
    from c05_reproduction import calculate_smiles
    from public_common import load_processor
    from public_models import trained_plan, complete_job
    from catboost import CatBoostRegressor
    run = safe_run(run_name)
    plan = trained_plan(run)
    require(not target.exists(), 'Output exists')
    frame = pd.read_csv(source,dtype=str,keep_default_na=False)
    require(frame.columns.tolist()==['row_id','Drug'] and len(frame)>0 and frame.row_id.is_unique and frame.row_id.str.strip().ne('').all(), 'Invalid input table')
    ids=frame.row_id.tolist()
    bits,raw=calculate_smiles(ids,frame.Drug.tolist())
    predictions=[]
    for seed in SEEDS:
        spec=next(j for j in plan['jobs'] if j['experiment']=='S32_CATBOOST' and j['split_seed']==seed)
        folder=run/'models'/spec['job_id']
        accepted=complete_job(folder)
        processor,state=load_processor(seed)
        desc=processor[:-1].transform(raw[:,state['available_indices']])
        x=np.concatenate([bits.astype(np.float64),desc],axis=1)
        col=2048+state['output_feature_names'].index('Ipc')
        require(np.isfinite(x).all() and (x[:,col]>=0).all(),'Invalid Ipc')
        x[:,col]=np.log1p(x[:,col])
        require(np.isfinite(x.astype(np.float32)).all(),'Tree input overflow')
        model=CatBoostRegressor()
        model.load_model(str(folder/accepted['model_file']))
        p=np.asarray(model.predict(x),dtype=float)
        require(p.shape==(len(ids),) and np.isfinite(p).all(),'Invalid predictions')
        predictions.append(p)
    mean=np.mean(predictions,axis=0)
    write_csv(target,[dict(row_id=rid,prediction_mean_tdc_scale=float(mean[i]),**{f'seed_{s}':float(predictions[s-1][i]) for s in SEEDS}) for i,rid in enumerate(ids)])
    print('Predictions exported; five-seed mean is descriptive, not an accepted ensemble benchmark score')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='stage',required=True)
    p=commands.add_parser('download',help='Get and hash-check official BenchmarkGroup data')
    p.add_argument('--cache',type=Path,help='Use a matching existing TDC cache instead of downloading')
    commands.add_parser('splits',help='Rebuild official seeds and verify exact membership/serialization')
    p=commands.add_parser('features',help='Recalculate raw representations without reading labels')
    p.add_argument('--part',choices=['train_val','test'],required=True)
    commands.add_parser('preprocessing',help='Fit five train-only processors')
    commands.add_parser('audit-data',help='Report raw/canonical/scaffold overlaps without reading label values')
    p=commands.add_parser('check',help='Check environment, exact primary matrices and GPU readiness')
    p.add_argument('--scope',choices=['primary','paper'],default='primary')
    p=commands.add_parser('validate',help='Run full frozen validation search; long GPU task')
    p.add_argument('--run',required=True)
    p.add_argument('--scope',choices=['primary','paper'],default='paper')
    p.add_argument('--resume',action='store_true')
    p=commands.add_parser('freeze',help='Freeze final jobs using historical selection or new validation only')
    p.add_argument('--run',required=True)
    p.add_argument('--scope',choices=['primary','paper'],default='primary')
    p.add_argument('--selection',choices=['historical','validation'],default='historical')
    p=commands.add_parser('train',help='Fit five or fifty fresh models on training subsets only')
    p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='full')
    p.add_argument('--resume',action='store_true')
    p=commands.add_parser('predict',help='Generate blind test predictions after all final fits')
    p.add_argument('--run',required=True)
    p.add_argument('--resume',action='store_true')
    p=commands.add_parser('evaluate',help='Read test labels only after the blind-prediction seal')
    p.add_argument('--run',required=True)
    p=commands.add_parser('report',help='Rebuild manuscript tables and figures from a completed run')
    p.add_argument('--run',required=True)
    p.add_argument('--validation-run',help='Use a complete new validation matrix instead of historical summaries')
    p=commands.add_parser('predict-smiles',help='Serve newly fitted primary models on new structures')
    p.add_argument('--run',required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Development stages have no permitted route to held-out labels or features.
    if args.stage in ('splits','preprocessing','check','validate','freeze','train') or args.stage=='features' and args.part=='train_val':
        development_guard()
    if args.stage in ('download','splits','features','preprocessing','audit-data'):
        import public_data as data
        if args.stage=='download':data.download(args.cache)
        elif args.stage=='splits':data.splits()
        elif args.stage=='features':data.features(args.part)
        elif args.stage=='preprocessing':data.preprocessing()
        else:data.audit_structure()
    elif args.stage=='check':check(args.scope)
    elif args.stage in ('validate','freeze','train','predict'):
        import public_models as models
        if args.stage=='validate':models.validate(args.run,args.scope,args.resume)
        elif args.stage=='freeze':models.freeze(args.run,args.scope,args.selection)
        elif args.stage=='train':models.train(args.run,args.mode,args.resume)
        else:models.predict(args.run,args.resume)
    elif args.stage in ('evaluate','report'):
        import public_reports as reports
        if args.stage=='evaluate':reports.evaluate(args.run)
        else:reports.report(args.run,args.validation_run)
    else:predict_smiles(args.run,args.input,args.output)


if __name__=='__main__':
    main()
