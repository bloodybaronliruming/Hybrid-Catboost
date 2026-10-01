"""Independent evaluation and manuscript tables from generated blind predictions."""
import json
import numpy as np
import pandas as pd
from public_common import (ROOT, DATA, SEEDS, require, sha, read, dump, recipe,
    safe_run, verify, split_frame, csv_rows, write_csv, metrics, input_manifest, stable_sd)
from public_models import trained_plan, complete_job


def summarize(records):
    result = []
    groups = sorted({r['experiment'] for r in records})
    for group in groups:
        runs = sorted((r for r in records if r['experiment'] == group), key=lambda r: int(r['seed']))
        require([int(r['seed']) for r in runs] == list(SEEDS), 'Five independent runs required')
        row = dict(experiment=group, candidate=runs[0].get('candidate'), seeds=list(SEEDS), test_rows_per_run=1997)
        for field in ('MAE', 'RMSE', 'R2', 'Spearman'):
            values = [r[field] for r in runs]
            require(all(v is None for v in values) or all(v is not None and np.isfinite(v) for v in values), 'Mixed/nonfinite metric')
            row[field + '_per_seed'] = values
            row[field + '_mean'] = None if values[0] is None else float(np.mean(values))
            row[field + '_sample_sd'] = None if values[0] is None else float(stable_sd(values, ddof=1))
            row[field + '_population_sd'] = None if values[0] is None else float(stable_sd(values, ddof=0))
        result.append(row)
    return result


def evaluated_predictions(run, plan):
    seal = read(run / 'blind_predictions_complete.json')
    require(seal['status'] == 'PASS' and seal['fits'] == len(plan['jobs']) and seal['final_plan_sha256'] == sha(run / 'final_plan.json'), 'Blind predictions incomplete or plan changed')
    verify(seal)
    source = DATA / 'raw/test.csv'
    require(sha(source) == recipe()['expected_raw']['test']['sha256'], 'Held-out source changed')
    # This is the first stage that converts held-out label cells to numbers.
    frame = pd.read_csv(source)
    ids = [f'solubility_aqsoldb:test:{i}' for i in range(len(frame))]
    predictions = {}
    for spec in plan['jobs']:
        table = pd.read_csv(run / 'predictions' / spec['job_id'] / 'predictions.csv', float_precision='round_trip')
        require(table.row_id.tolist() == ids and len(table) == 1997, 'Prediction row order or coverage changed')
        values = table.prediction_tdc_scale.to_numpy(dtype=float)
        require(np.isfinite(values).all(), 'Nonfinite prediction')
        predictions[spec['job_id']] = values
    return frame.Y.to_numpy(dtype=float), ids, predictions


def evaluate(run_name):
    run = safe_run(run_name)
    plan = trained_plan(run)
    require(not (run / 'evaluation').exists(), 'Evaluation output exists; preserve it')
    labels, ids, predictions = evaluated_predictions(run, plan)
    records, comparisons = [], []
    original = {r['experiment']: r for r in read(ROOT / 'results/manuscript_metrics_lock.json')['all_groups_full_precision']}
    for spec in plan['jobs']:
        measured = metrics(labels, predictions[spec['job_id']])
        record = dict(experiment=spec['experiment'], candidate=spec['candidate'], seed=spec['split_seed'], rows=len(ids),
                      **measured, Spearman_reason='undefined for constant vector' if measured['Spearman'] is None else '')
        records.append(record)
        tolerance = recipe()['metrics_tolerance']['dmpnn' if spec['algorithm'] == 'dmpnn' else 'traditional']
        for field, value in measured.items():
            expected = original[spec['experiment']][field + '_per_seed'][spec['split_seed'] - 1]
            equal = value is None and expected is None if value is None or expected is None else bool(np.isclose(value, expected, **tolerance))
            comparisons.append(dict(experiment=spec['experiment'], seed=spec['split_seed'], metric=field,
                measured=value, historical=expected, within_tolerance=equal, **tolerance))
    summary = summarize(records)
    primary = next(r for r in summary if r['experiment'] == 'S32_CATBOOST')
    expected_primary = original['S32_CATBOOST']
    allowed = recipe()['primary_refit_mae_tolerance']
    primary_mae_ok = all(abs(x-y) <= allowed['per_seed_atol'] for x,y in zip(primary['MAE_per_seed'], expected_primary['MAE_per_seed'])) and abs(primary['MAE_mean']-expected_primary['MAE_mean']) <= allowed['mean_atol']
    native = {}
    for row in summary:
        native[row['experiment']] = {}
        for field in ('MAE', 'RMSE', 'R2', 'Spearman'):
            values = row[field + '_per_seed']
            rounded = None if values[0] is None else [round(v, 3) for v in values]
            native[row['experiment']][field] = None if rounded is None else dict(individual_round3=rounded,
                mean_population_sd_round3=[round(float(np.mean(rounded)),3), round(float(stable_sd(rounded,ddof=0)),3)])
    folder = run / 'evaluation'
    write_csv(folder / 'per_seed_metrics.csv', records)
    write_csv(folder / 'historical_comparisons.csv', comparisons)
    dump(folder / 'summary.json', dict(status='COMPUTED', all_groups_full_precision=summary, native_TDC=native,
        strict_historical_metric_match=all(x['within_tolerance'] for x in comparisons),
        primary_R3_MAE_tolerance_pass=primary_mae_ok, primary_R3_tolerances=allowed,
        historical_selection_differences=plan['historical_selection_differences'],
        new_external_validation=False, prediction_ensemble_scored=False))
    dump(folder / 'manifest.json', input_manifest(folder, list(folder.iterdir()), test_source_sha256=sha(DATA / 'raw/test.csv')))
    print(json.dumps(dict(status='COMPUTED', strict_historical_metric_match=all(x['within_tolerance'] for x in comparisons), primary_R3_MAE_tolerance_pass=primary_mae_ok)))


def validation_records(run, plan, validation_run_name):
    if validation_run_name is None:
        records = []
        for row in csv_rows(ROOT / 'evidence/historical_validation_runs.csv'):
            records.append(dict(experiment=row['experiment'], seed=int(row['seed']),
                **{key: None if row[key] == '' else float(row[key]) for key in ('MAE','RMSE','R2','Spearman')}))
        return records, 'historical selected validation metrics; not rerun in this report'
    source = safe_run(validation_run_name)
    vp = read(source / 'validation_plan.json')
    from public_models import verify_binding
    verify_binding(vp['binding'], vp['scope'])
    require(read(source / 'validation_complete.json')['fits'] == len(vp['jobs']), 'Validation matrix incomplete')
    records = []
    for spec in plan['jobs']:
        folder = source / 'validation' / f"{spec['experiment']}__{spec['candidate']}__seed{spec['split_seed']}"
        record = complete_job(folder)
        records.append(dict(experiment=spec['experiment'], seed=spec['split_seed'], **record['metrics']))
    return records, 'new complete validation run: ' + validation_run_name


def report(run_name, validation_run_name=None):
    run = safe_run(run_name)
    plan = trained_plan(run)
    verify(read(run / 'evaluation/manifest.json'))
    folder = run / 'report'
    require(not folder.exists(), 'Report output exists; preserve it')
    labels, ids, predictions = evaluated_predictions(run, plan)
    computed = read(run / 'evaluation/summary.json')
    records = csv_rows(run / 'evaluation/per_seed_metrics.csv')
    validation, validation_origin = validation_records(run, plan, validation_run_name)
    selected_groups = {j['experiment'] for j in plan['jobs']}
    validation = [r for r in validation if r['experiment'] in selected_groups]
    validation_summary = summarize(validation)
    write_csv(folder / 'data/validation_summary.csv', [{k:v for k,v in r.items() if k in ('experiment','MAE_mean','MAE_sample_sd','RMSE_mean','RMSE_sample_sd','R2_mean','R2_sample_sd','Spearman_mean','Spearman_sample_sd')} for r in validation_summary])
    lookup = {(r['experiment'],r['seed']):r for r in validation}
    paired = []
    pairs = read(ROOT / 'results/manuscript_metrics_lock.json')['paired_validation_summary']
    for pair in pairs:
        if pair['baseline'] not in selected_groups or pair['comparison'] not in selected_groups:
            continue
        deltas = [lookup[pair['comparison'],s]['MAE']-lookup[pair['baseline'],s]['MAE'] for s in SEEDS]
        paired.append(dict(baseline=pair['baseline'], comparison=pair['comparison'], n_runs=5,
            MAE_delta_mean=float(np.mean(deltas)), MAE_delta_sample_sd=float(np.std(deltas,ddof=1)),
            MAE_improved_seeds=int(np.sum(np.array(deltas)<0))))
    thresholds, quartile_runs, provenance, quartile_counts = [], [], [], []
    for seed in SEEDS:
        train = split_frame(seed, 'train')
        q = np.quantile(train.Y.to_numpy(), [.25,.5,.75], method='linear')
        thresholds.append(dict(seed=seed, train_rows=len(train), q25=float(q[0]),q50=float(q[1]),q75=float(q[2]),method='linear; train labels only'))
        bins = np.searchsorted(q, labels, side='left') + 1
        quartile_counts += [dict(seed=seed,quartile=b,fixed_test_records=int((bins==b).sum()),
            threshold_origin="corresponding seed's training-label linear quartiles") for b in range(1,5)]
        for spec in (j for j in plan['jobs'] if j['split_seed']==seed):
            values = predictions[spec['job_id']]
            for bin_index in range(1,5):
                mask = bins == bin_index
                require(mask.any(), 'Empty quartile')
                residual = values[mask]-labels[mask]
                quartile_runs.append(dict(experiment=spec['experiment'],seed=seed,bin=bin_index,rows=int(mask.sum()),
                    MAE=float(np.mean(np.abs(residual))),RMSE=float(np.sqrt(np.mean(residual**2))),signed_bias=float(np.mean(residual))))
        spec = next(j for j in plan['jobs'] if j['experiment']=='S32_CATBOOST' and j['split_seed']==seed)
        accepted = complete_job(run / 'models' / spec['job_id'])
        provenance.append(dict(seed=seed,train_records=len(train),validation_records=len(split_frame(seed,'valid')),fixed_test_records=1997,
            candidate=spec['candidate'],config_sha256=sha(run/'final_plan.json'),processor_sha256=sha(DATA/f'process/processors/seed_{seed}/processor.joblib'),
            model_sha256=accepted['model_sha256'],prediction_sha256=sha(run/'predictions'/spec['job_id']/'predictions.csv'),
            provenance_scope='new locally generated artifacts; not historical weights'))
    quartiles = []
    for group in sorted(selected_groups):
        for bin_index in range(1,5):
            rr = [r for r in quartile_runs if r['experiment']==group and r['bin']==bin_index]
            row = dict(experiment=group, bin=bin_index, n_runs=5)
            for metric in ('MAE','RMSE','signed_bias'):
                values = [r[metric] for r in rr]
                row[metric+'_mean'] = float(np.mean(values))
                row[metric+'_sample_sd'] = float(np.std(values,ddof=1))
            quartiles.append(row)
    primary = np.vstack([predictions[f'S32_CATBOOST__seed{s}'] for s in SEEDS]).mean(axis=0)
    write_csv(folder/'data/final_compound_summary.csv',[dict(experiment='S32_CATBOOST',row_id=rid,Y_true=float(y),prediction_mean=float(p)) for rid,y,p in zip(ids,labels,primary)])
    write_csv(folder/'data/target_range_performance.csv',quartile_runs)
    write_csv(folder/'data/target_range_performance_summary.csv',quartiles)
    write_csv(folder/'data/target_range_thresholds.csv',thresholds)
    write_csv(folder/'data/paired_validation_summary.csv',paired, fields=list(paired[0]) if paired else ['baseline','comparison','n_runs','MAE_delta_mean','MAE_delta_sample_sd','MAE_improved_seeds'])
    write_csv(folder/'data/per_seed_metrics.csv',records)
    write_csv(folder/'data/s32_seed_provenance.csv',provenance)
    from public_diagnostics import diagnostics
    diagnostics(folder,labels,ids,predictions,plan)
    dump(folder/'data/plot_metrics.json',dict(status='COMPUTED',public_model_name='Hybrid-CatBoost',internal_group='S32_CATBOOST',seeds=list(SEEDS),test_rows_per_run=1997,
        all_groups_full_precision=computed['all_groups_full_precision'],paired_validation_summary=paired))
    if plan['scope']=='paper':
        from public_plot import render
        render(folder)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(180/25.4,85/25.4),layout='constrained')
        axes[0].scatter(labels,primary,s=6,alpha=.5,color='#009E73',rasterized=True)
        lo,hi=min(labels.min(),primary.min()),max(labels.max(),primary.max())
        axes[0].plot([lo,hi],[lo,hi],'--',color='#444444')
        axes[0].set(xlabel='Observed TDC logS',ylabel='Mean predicted TDC logS')
        axes[1].scatter(labels,primary-labels,s=6,alpha=.5,color='#009E73',rasterized=True)
        axes[1].axhline(0,ls='--',color='#444444')
        axes[1].set(xlabel='Observed TDC logS',ylabel='Mean signed residual')
        (folder/'figures').mkdir()
        for ext in ('pdf','svg','png'):
            fig.savefig(folder/f'figures/primary_diagnostics.{ext}',dpi=600)
        plt.close(fig)
    from public_pipeline_figure import render_pipeline
    render_pipeline(folder/'figures')
    write_csv(folder/'tables/table_s04_quartile_counts.csv',quartile_counts)
    write_csv(folder/'tables/table_s05_s32_seed_provenance.csv',provenance)
    files=[p for p in folder.rglob('*') if p.is_file()]
    dump(folder/'manifest.json',input_manifest(folder,files,status='GENERATED_PENDING_VISUAL_REVIEW',
        validation_origin=validation_origin,strict_historical_metric_match=computed['strict_historical_metric_match'],
        source_evaluation_sha256=sha(run/'evaluation/manifest.json'),descriptive_only=True,ensemble_benchmark_score=False))
    print('Report generated; validation origin:',validation_origin)
