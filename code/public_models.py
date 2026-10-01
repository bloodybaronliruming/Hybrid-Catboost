"""Validation, frozen fitting and blind inference without private model artifacts."""
import copy
import time
from public_common import (ROOT, DATA, PROCESS, SEEDS, require, sha, read, dump,
    recipe, safe_run, split_frame, raw_blocks, view, metrics, write_csv,
    check_environment, source_binding, verify, input_manifest, ordered_hash, matrix_sha)

GRAPH_CACHE = {}


def estimator(algorithm, parameters):
    if algorithm in ('mean', 'median'):
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(**parameters)
    if algorithm == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**parameters)
    if algorithm == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**parameters)
    if algorithm == 'catboost':
        from catboost import CatBoostRegressor
        from catboost.utils import get_gpu_device_count
        require(get_gpu_device_count() > 0, 'GPU required for frozen CatBoost fitting')
        return CatBoostRegressor(**parameters)
    raise ValueError('Unknown learner')


def graph_rows(part, frame=None, labels=False):
    import pandas as pd
    import public_graph as graph
    if part not in GRAPH_CACHE:
        source = DATA / f'raw/{part}.csv'
        require(sha(source) == recipe()['expected_raw'][part]['sha256'], 'Graph source changed')
        smiles = pd.read_csv(source, usecols=['Drug'], dtype=str, keep_default_na=False).Drug.tolist()
        GRAPH_CACHE[part] = [graph.graph(text) for text in smiles]
    graphs = GRAPH_CACHE[part]
    if frame is None:
        ids = [f'solubility_aqsoldb:{part}:{i}' for i in range(len(graphs))]
        return graph.GraphRows(graphs, ids)
    selected = frame.source_row_index.to_numpy(dtype=int)
    require(frame.row_id.tolist() == [f'solubility_aqsoldb:{part}:{i}' for i in selected], 'Graph row identity differs')
    return graph.GraphRows([graphs[i] for i in selected], frame.row_id.tolist(), frame.Y.to_numpy() if labels else None)


def binding(scope):
    graph = scope == 'paper'
    environment = check_environment(graph)
    paths = [PROCESS / 'splits/manifest.json', PROCESS / 'features/train_val/manifest.json']
    paths += [PROCESS / f'processors/seed_{seed}/manifest.json' for seed in SEEDS]
    for path in paths:
        verify(read(path))
    return dict(source_sha256=source_binding(), environment=environment,
                input_manifests_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})


def verify_binding(record, scope):
    require(record == binding(scope), 'Source, environment or development inputs changed; resume refused')


def fit_one(spec, folder, validation=False, smoke=False):
    import numpy as np
    from threadpoolctl import threadpool_limits
    train = split_frame(spec['split_seed'], 'train')
    valid = split_frame(spec['split_seed'], 'valid') if validation else None
    if smoke:
        train = train.iloc[:32].copy()
        if valid is not None:
            valid = valid.iloc[:8].copy()
    params = dict(spec['parameters'])
    if smoke:
        if spec['algorithm'] == 'catboost':
            params['iterations'] = 2
        elif spec['algorithm'] == 'random_forest':
            params.update(n_estimators=2, max_depth=3)
        elif spec['algorithm'] == 'dmpnn':
            params['max_epochs'] = 1
    start = time.monotonic()
    details = dict(fit_rows=len(train), fit_row_ids_sha256=ordered_hash(train.row_id.tolist()), mode='smoke' if smoke else 'full')
    if spec['algorithm'] == 'dmpnn':
        import public_graph as graph
        graph.setup(spec['split_seed'], 'cuda:0')
        net = graph.model(params).to('cuda:0')
        dataset = graph_rows('train_val', train, labels=True)
        if validation:
            values, history = graph.fit(net, dataset, graph_rows('train_val', valid), valid.Y.to_numpy(), params, 'cuda:0', folder)
            details['best_epoch'] = history['best_epoch']
            details['epochs_run'] = history['epochs_run']
            dump(folder / 'training_history.json', history)
            model_file = folder / 'best_model.pt'
        else:
            history = graph.fit_graph_fixed(net, dataset, params, 'cuda:0', out=folder)
            details['epochs_run'] = history['epochs_run']
            model_file = folder / 'final_model.pt'
            values = None
    else:
        import joblib
        index = train.source_row_index.to_numpy(dtype=int)
        x, names = view(spec, 'train_val', index)
        details.update(matrix_sha256=matrix_sha(x), feature_columns=x.shape[1])
        if 'matrix_sha256' in spec and not smoke:
            require(details['matrix_sha256'] == spec['matrix_sha256'], 'Historical primary training matrix differs')
        if 'descriptor_names' in spec and spec['representation'] in ('RDKit2D', 'Hybrid'):
            descriptor_names = names[2048:] if spec['representation'] == 'Hybrid' else names
            require(descriptor_names == spec['descriptor_names'], 'Frozen feature order differs')
        model = estimator(spec['algorithm'], params)
        with threadpool_limits(limits=1):
            model.fit(x, train.Y.to_numpy(dtype=np.float64))
        model_file = folder / ('model.cbm' if spec['algorithm'] == 'catboost' else 'model.joblib')
        if spec['algorithm'] == 'catboost':
            model.save_model(str(model_file))
        else:
            joblib.dump(model, model_file)
        if validation:
            vx, _ = view(spec, 'train_val', valid.source_row_index.to_numpy(dtype=int))
            with threadpool_limits(limits=1):
                values = model.predict(vx)
        else:
            values = None
    details.update(status='PASS', model_file=model_file.name, model_sha256=sha(model_file), seconds=time.monotonic()-start)
    if validation:
        details['metrics'] = metrics(valid.Y.to_numpy(), values)
        write_csv(folder / 'validation_predictions.csv', [dict(row_id=rid, prediction_tdc_scale=float(p)) for rid, p in zip(valid.row_id, values)])
        details['validation_predictions_sha256'] = sha(folder / 'validation_predictions.csv')
    return details


def complete_job(folder):
    record = read(folder / 'acceptance.json')
    require(record['status'] == 'PASS' and sha(folder / record['model_file']) == record['model_sha256'], 'Unaccepted or changed model')
    if 'validation_predictions_sha256' in record:
        require(sha(folder / 'validation_predictions.csv') == record['validation_predictions_sha256'], 'Changed validation predictions')
    return record


def execute_jobs(directory, jobs, scope, validation=False, smoke=False, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    status = []
    for spec in jobs:
        folder = directory / spec['job_id']
        if folder.exists():
            require(resume and (folder / 'acceptance.json').is_file(), 'Existing incomplete/failed job; preserve it and start a fresh run: ' + spec['job_id'])
            require(read(folder / 'specification.json') == spec, 'Resume specification differs')
            status.append(complete_job(folder))
            continue
        folder.mkdir()
        dump(folder / 'specification.json', spec)
        try:
            record = fit_one(spec, folder, validation, smoke)
            record.update(job_id=spec['job_id'], experiment=spec['experiment'], candidate=spec['candidate'], seed=spec['split_seed'])
            dump(folder / 'acceptance.json', record)
            status.append(record)
            print(spec['job_id'], 'PASS', flush=True)
        except Exception as error:
            dump(folder / 'failure.json', dict(status='FAIL', job_id=spec['job_id'], error_type=type(error).__name__, error=str(error)))
            raise
    return status


def validation_jobs(scope):
    r = recipe()
    jobs = []
    groups = [e for e in r['experiments'] if scope == 'paper' or e['id'] == r['primary']]
    for group in groups:
        for candidate in r['candidates'][group['algorithm']]:
            for seed in SEEDS:
                parameters = {**r['fixed_parameters'].get(group['algorithm'], {}), **candidate['parameters']}
                if group['algorithm'] == 'random_forest':
                    parameters['random_state'] = seed
                if group['algorithm'] == 'catboost':
                    parameters['random_seed'] = seed
                if group['algorithm'] == 'dmpnn':
                    parameters['seed'] = seed
                jobs.append(dict(job_id=f"{group['id']}__{candidate['candidate_id']}__seed{seed}",
                    experiment=group['id'], candidate=candidate['candidate_id'], algorithm=group['algorithm'],
                    representation=group['representation'], split_seed=seed, parameters=parameters))
    require(len(jobs) == (465 if scope == 'paper' else 60), 'Validation matrix differs')
    return jobs


def validate(run_name, scope, resume=False):
    run = safe_run(run_name)
    plan = dict(scope=scope, binding=binding(scope), jobs=validation_jobs(scope))
    if (run / 'validation_plan.json').exists():
        require(resume and read(run / 'validation_plan.json') == plan, 'Validation plan differs or resume not requested')
    else:
        require(not run.exists(), 'Run already exists')
        dump(run / 'validation_plan.json', plan)
    records = execute_jobs(run / 'validation', plan['jobs'], scope, validation=True, resume=resume)
    if not (run / 'validation_complete.json').exists():
        dump(run / 'validation_complete.json', dict(status='PASS', fits=len(records), binding=plan['binding']))
    print('PASS: complete validation matrix; no test data accessed')


def select_validation(run, scope):
    import numpy as np
    plan = read(run / 'validation_plan.json')
    require(plan['scope'] == scope, 'Scope differs')
    verify_binding(plan['binding'], scope)
    require(read(run / 'validation_complete.json')['fits'] == len(plan['jobs']), 'Validation is incomplete')
    groups = [e for e in recipe()['experiments'] if scope == 'paper' or e['id'] == recipe()['primary']]
    jobs, selection = [], []
    for group in groups:
        candidates = []
        for order, candidate in enumerate(recipe()['candidates'][group['algorithm']]):
            records = [complete_job(run / 'validation' / f"{group['id']}__{candidate['candidate_id']}__seed{s}") for s in SEEDS]
            values = [r['metrics']['MAE'] for r in records]
            require(np.isfinite(values).all(), 'Nonfinite validation MAE; selection blocked')
            candidates.append((float(np.mean(values)), float(np.std(values, ddof=1)), order, candidate, records))
        mean, sd, order, candidate, records = min(candidates, key=lambda x: x[:3])
        selection.append(dict(experiment=group['id'], candidate=candidate['candidate_id'], MAE_mean=mean, MAE_sample_sd=sd))
        for seed, record in zip(SEEDS, records):
            spec = next(j for j in plan['jobs'] if j['experiment'] == group['id'] and j['candidate'] == candidate['candidate_id'] and j['split_seed'] == seed)
            spec = copy.deepcopy(spec)
            spec['job_id'] = f"{group['id']}__seed{seed}"
            if group['algorithm'] == 'dmpnn':
                spec['parameters']['max_epochs'] = record['best_epoch']
                del spec['parameters']['patience']
            jobs.append(spec)
    return jobs, selection


def freeze(run_name, scope, selection='historical'):
    run = safe_run(run_name)
    if selection == 'validation':
        jobs, choices = select_validation(run, scope)
    else:
        require(not run.exists(), 'Historical freeze requires a fresh run name')
        jobs = [j for j in recipe()['historical_final'] if scope == 'paper' or j['experiment'] == recipe()['primary']]
        choices = [dict(experiment=j['experiment'], candidate=j['candidate']) for j in jobs if j['split_seed'] == 1]
    historical = {j['job_id']: j for j in recipe()['historical_final']}
    differences = []
    for j in jobs:
        h = historical[j['job_id']]
        for invariant in ('descriptor_names', 'feature_columns', 'matrix_sha256'):
            if invariant in h:
                j[invariant] = h[invariant]
        if j['candidate'] != h['candidate'] or j['parameters'] != h['parameters']:
            differences.append(j['job_id'])
    dump(run / 'final_plan.json', dict(scope=scope, selection_source=selection, binding=binding(scope),
         jobs=jobs, selections=choices, historical_selection_differences=differences,
         test_used_for_selection=False, early_stopping_in_final_fit=False))
    print('FROZEN:', len(jobs), 'independent final fits;', len(differences), 'historical selection differences')


def train(run_name, mode='full', resume=False):
    run = safe_run(run_name)
    plan = read(run / 'final_plan.json')
    verify_binding(plan['binding'], plan['scope'])
    jobs = plan['jobs'] if mode == 'full' else [plan['jobs'][0]]
    records = execute_jobs(run / ('models' if mode == 'full' else 'smoke'), jobs, plan['scope'], smoke=mode == 'smoke', resume=resume)
    marker = run / ('training_complete.json' if mode == 'full' else 'smoke_complete.json')
    if not marker.exists():
        dump(marker, dict(status='PASS' if mode == 'full' else 'PASS_SMOKE', fits=len(records), plan_sha256=sha(run / 'final_plan.json'), binding=plan['binding']))
    print('PASS:', mode, len(records), 'fits; no held-out data accessed')


def trained_plan(run):
    plan = read(run / 'final_plan.json')
    verify_binding(plan['binding'], plan['scope'])
    complete = read(run / 'training_complete.json')
    require(complete['status'] == 'PASS' and complete['fits'] == len(plan['jobs']) and complete['plan_sha256'] == sha(run / 'final_plan.json'), 'All frozen fits must complete first')
    for job in plan['jobs']:
        complete_job(run / 'models' / job['job_id'])
    return plan


def predict_model(spec, folder, part, frame=None):
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    record = complete_job(folder)
    if spec['algorithm'] == 'dmpnn':
        import torch
        import public_graph as graph
        graph.setup(spec['split_seed'], 'cuda:0')
        net = graph.model(spec['parameters']).to('cuda:0')
        net.load_state_dict(torch.load(folder / record['model_file'], map_location='cuda:0', weights_only=True))
        dataset = graph_rows(part, frame)
        values = graph.predict(net, dataset, spec['parameters']['batch_size'], 'cuda:0')
        ids = dataset.row_ids
    else:
        positions = None if frame is None else frame.source_row_index.to_numpy(dtype=int)
        x, _ = view(spec, part, positions)
        ids, _, _ = raw_blocks(part)
        if positions is not None:
            ids = [ids[i] for i in positions]
        if spec['algorithm'] == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(folder / record['model_file']))
        else:
            model = joblib.load(folder / record['model_file'])
        with threadpool_limits(limits=1):
            values = np.asarray(model.predict(x), dtype=np.float64)
    require(values.shape == (len(ids),) and np.isfinite(values).all(), 'Invalid prediction coverage')
    return ids, values


def predict(run_name, resume=False):
    run = safe_run(run_name)
    plan = trained_plan(run)
    raw_blocks('test')
    records = []
    for spec in plan['jobs']:
        folder = run / 'predictions' / spec['job_id']
        if folder.exists():
            require(resume and (folder / 'acceptance.json').exists(), 'Prediction output exists')
            record = read(folder / 'acceptance.json')
            require(sha(folder / 'predictions.csv') == record['predictions_sha256'], 'Predictions changed')
        else:
            ids, values = predict_model(spec, run / 'models' / spec['job_id'], 'test')
            write_csv(folder / 'predictions.csv', [dict(row_id=rid, prediction_tdc_scale=float(p)) for rid, p in zip(ids, values)])
            record = dict(status='PASS', rows=len(ids), predictions_sha256=sha(folder / 'predictions.csv'))
            dump(folder / 'acceptance.json', record)
        records.append(record)
    if not (run / 'blind_predictions_complete.json').exists():
        files = [run / 'predictions' / j['job_id'] / 'predictions.csv' for j in plan['jobs']]
        dump(run / 'blind_predictions_complete.json', input_manifest(run, files, status='PASS', fits=len(records),
             final_plan_sha256=sha(run / 'final_plan.json'), test_labels_loaded=False))
    print('PASS: all blind fixed-test predictions; labels were not read')
