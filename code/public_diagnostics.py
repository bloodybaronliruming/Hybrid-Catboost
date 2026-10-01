"""Descriptive difficult-record and structural-domain analyses; no model changes."""
import numpy as np
from public_common import ROOT, DATA, require, read, dump, write_csv, split_frame, raw_blocks, load_processor, stable_sd


def diagnostics(folder, labels, ids, predictions, plan):
    from rdkit import DataStructs
    dev_ids, dev_bits, dev_desc = raw_blocks('train_val')
    test_ids, test_bits, test_desc = raw_blocks('test')
    require(ids == test_ids, 'Diagnostic test identities differ')
    fingerprints = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in np.packbits(dev_bits,axis=1,bitorder='little')]
    queries = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in np.packbits(test_bits,axis=1,bitorder='little')]
    states = {}
    for seed in range(1,6):
        frame = split_frame(seed,'train')
        positions = frame.source_row_index.to_numpy(dtype=int)
        _, state = load_processor(seed)
        selected = state['selected_indices']
        train_values = dev_desc[np.ix_(positions,selected)]
        finite = np.isfinite(train_values)
        minimum = np.min(np.where(finite,train_values,np.inf),axis=0)
        maximum = np.max(np.where(finite,train_values,-np.inf),axis=0)
        require(np.isfinite(minimum).all() and np.isfinite(maximum).all(),'Missing selected training descriptor')
        values = test_desc[:,selected]
        valid = np.isfinite(values)
        outside = valid & ((values<minimum)|(values>maximum))
        states[seed] = dict(positions=positions,selected=selected,outside=outside.sum(axis=1),missing=(~valid).sum(axis=1),nearest=[])
    neighbors=[]
    for i,query in enumerate(queries):
        scores=np.asarray(DataStructs.BulkTanimotoSimilarity(query,fingerprints))
        for seed,state in states.items():
            winner=int(np.argmax(scores[state['positions']]))
            position=int(state['positions'][winner])
            maximum=float(scores[position])
            state['nearest'].append(maximum)
            neighbors.append(dict(seed=seed,row_id=ids[i],max_tanimoto=maximum,nearest_train_row_id=dev_ids[position],
                descriptors_outside_range=int(state['outside'][i]),descriptors_missing=int(state['missing'][i])))
    write_csv(folder/'data/applicability_domain_records.csv',neighbors)
    groups=sorted({j['experiment'] for j in plan['jobs']})
    difficult,disagreement,shrinkage=[],[],[]
    from scipy.stats import spearmanr
    for group in groups:
        values=np.vstack([predictions[f'{group}__seed{s}'] for s in range(1,6)])
        residual=values-labels
        means=values.mean(axis=0)
        mean_absolute=np.abs(residual).mean(axis=0)
        deviations=stable_sd(values,axis=0)
        # Ties use row_id lexical order, as in the original post-test analysis.
        ranking=sorted(range(len(ids)),key=lambda i:(-mean_absolute[i],ids[i]))
        for rank,i in enumerate(ranking,1):
            difficult.append(dict(experiment=group,row_id=ids[i],Y_true=float(labels[i]),n_seeds=5,
                prediction_mean=float(means[i]),prediction_sample_sd=float(deviations[i]),
                mean_absolute_residual=float(mean_absolute[i]),absolute_residual_of_mean_prediction=float(abs(means[i]-labels[i])),
                error_rank=rank,top15=rank<=15,top50=rank<=50,
                similarity_mean=float(np.mean([states[s]['nearest'][i] for s in range(1,6)])),
                descriptor_outside_seed_count=sum(int(states[s]['outside'][i]>0) for s in range(1,6)),
                descriptor_missing_seed_count=sum(int(states[s]['missing'][i]>0) for s in range(1,6))))
        disagreement.append(dict(experiment=group,rows=len(ids),Spearman=None if np.ptp(deviations)==0 or np.ptp(mean_absolute)==0 else float(spearmanr(deviations,mean_absolute).statistic),
            interpretation='seed disagreement is not calibrated uncertainty'))
        for seed,p in enumerate(values,1):
            centered=labels-labels.mean()
            slope=float(np.dot(centered,p-p.mean())/np.dot(centered,centered))
            shrinkage.append(dict(experiment=group,seed=seed,prediction_sample_sd=float(stable_sd(p)),
                observation_sample_sd=float(stable_sd(labels)),sd_ratio=float(stable_sd(p)/stable_sd(labels)),
                descriptive_ols_slope=slope,descriptive_ols_intercept=float(p.mean()-slope*labels.mean())))
    write_csv(folder/'data/difficult_records.csv',difficult)
    write_csv(folder/'data/prediction_disagreement_correlations.csv',disagreement)
    write_csv(folder/'data/shrinkage_diagnostics.csv',shrinkage)
    dump(folder/'data/diagnostic_scope.json',dict(post_test_descriptive=True,model_changes=False,
        mean_absolute_residual='mean of five absolute residuals, not error of an ensemble prediction',
        descriptor_domain='raw retained descriptor finite training min/max; missing cells counted separately',
        uncertainty_coverage_claim=False,source_label_error_claim=False))
