"""Acquire pinned official data, preserve source rows and fit train-only processors."""
from pathlib import Path
import csv
import importlib.metadata as metadata
import shutil
from unittest.mock import patch
from public_common import (ROOT, DATA, PROCESS, SEEDS, require, sha, read, dump,
                           recipe, ordered_hash, write_csv, input_manifest,
                           check_environment, split_frame, raw_blocks, fit_processor)


def check_tdc_source():
    distribution = metadata.distribution('PyTDC')
    for relative, expected in recipe()['pytdc_source_sha256'].items():
        require(sha(distribution.locate_file(relative)) == expected, 'PyTDC source differs: ' + relative)


def download(cache=None):
    import pandas as pd
    from tdc.benchmark_group import admet_group
    check_environment()
    check_tdc_source()
    raw = DATA / 'raw'
    require(not (raw / 'manifest.json').exists(), 'Acquisition already exists; preserve it')
    destination = raw / 'tdc_cache'
    if cache is not None:
        cache = Path(cache).resolve()
        # Offline mode copies only the endpoint's two official files, not the group archive.
        for part in ('train_val', 'test'):
            source = cache / f'admet_group/solubility_aqsoldb/{part}.csv'
            expected = recipe()['expected_raw'][part]['sha256']
            require(source.is_file() and sha(source) == expected, 'Offline cache is absent or differs')
            target = destination / f'admet_group/solubility_aqsoldb/{part}.csv'
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), 'Cache output exists')
            shutil.copyfile(source, target)
    group = admet_group(path=str(destination))
    benchmark = group.get(recipe()['requested_name'])
    require(benchmark['name'] == recipe()['dataset'], 'Wrong benchmark')
    outputs = []
    for part in ('train_val', 'test'):
        source = destination / f'admet_group/solubility_aqsoldb/{part}.csv'
        expected = recipe()['expected_raw'][part]
        require(sha(source) == expected['sha256'], 'Official data version differs; stop without updating the recipe')
        require(len(benchmark[part]) == expected['rows'] and list(benchmark[part].columns) == expected['columns'], 'Official data shape differs')
        pd.testing.assert_frame_equal(benchmark[part], pd.read_csv(source))
        target = raw / f'{part}.csv'
        require(not target.exists(), 'Raw output exists')
        shutil.copyfile(source, target)
        outputs += [source, target]
    dump(raw / 'manifest.json', input_manifest(raw, outputs, dataset=benchmark['name'],
         origin=recipe()['source_url'], label_transform='identity', source='official BenchmarkGroup files'))
    print('PASS: official acquisition; no label-based analysis or model fitting')


def identities(part):
    with (DATA / f'raw/{part}.csv').open(newline='') as stream:
        source = list(csv.DictReader(stream))
    return [dict(row_id=f'solubility_aqsoldb:{part}:{i}', source_partition=part,
                 source_row_index=i, **row) for i, row in enumerate(source)]


def splits():
    import pandas as pd
    from tdc.benchmark_group import admet_group
    check_environment()
    check_tdc_source()
    folder = PROCESS / 'splits'
    require(not folder.exists(), 'Split output already exists')
    source_path = (DATA / 'raw/tdc_cache/admet_group/solubility_aqsoldb/train_val.csv').resolve()
    require(sha(source_path) == recipe()['expected_raw']['train_val']['sha256'], 'Development source changed')
    source = pd.read_csv(source_path)
    originals = identities('train_val')
    group = admet_group(path=str(DATA / 'raw/tdc_cache'))
    original_read = pd.read_csv
    records, files = {}, []
    marker = '__source_position_for_identity__'
    for seed in SEEDS:
        calls = []
        def attach_positions(path, *args, **kwargs):
            require(Path(path).resolve() == source_path, 'Official splitter read unexpected data')
            frame = original_read(path, *args, **kwargs)
            pd.testing.assert_frame_equal(frame, source)
            frame[marker] = range(len(frame))
            calls.append(path)
            return frame
        with patch.object(pd, 'read_csv', side_effect=attach_positions):
            train, valid = group.get_train_valid_split(benchmark=recipe()['dataset'], split_type='default', seed=seed)
        require(len(calls) == 1, 'Unexpected splitter read count')
        records[str(seed)] = {}
        positions = []
        for part, frame in [('train', train), ('valid', valid)]:
            index = frame[marker].to_numpy(dtype=int)
            pd.testing.assert_frame_equal(frame.drop(columns=marker), source.iloc[index].reset_index(drop=True))
            rows = [dict(**originals[int(i)], seed=seed, split=part) for i in index]
            expected = recipe()['expected_splits'][str(seed)][part]
            require(len(rows) == expected['rows'] and ordered_hash([r['row_id'] for r in rows]) == expected['row_id_sha256'], 'Historical split membership differs')
            path = folder / f'seed_{seed}/{part}.csv'
            write_csv(path, rows)
            require(sha(path) == expected['csv_sha256'], 'Historical split serialization differs')
            records[str(seed)][part] = expected
            files.append(path)
            positions += index.tolist()
        require(sorted(positions) == list(range(len(source))), 'Split coverage or overlap differs')
    dump(folder / 'manifest.json', input_manifest(folder, files, splits=records, method='unmodified PyTDC default scaffold split', test_labels_read=False))
    print('PASS: five exact historical training/validation splits')


def features(part):
    import numpy as np
    from c05_reproduction import calculate_smiles
    check_environment()
    folder = PROCESS / f'features/{part}'
    require(not folder.exists(), 'Feature output exists')
    source = DATA / f'raw/{part}.csv'
    require(sha(source) == recipe()['expected_raw'][part]['sha256'], 'Raw data changed')
    # Projection reads structural cells only; Y is never converted or exposed.
    with source.open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = header.index('Drug')
        smiles = [row[index] for row in reader]
    ids = [f'solubility_aqsoldb:{part}:{i}' for i in range(len(smiles))]
    bits, desc = calculate_smiles(ids, smiles)
    folder.mkdir(parents=True)
    files = []
    for name, array in [('morgan', bits), ('rdkit2d', desc)]:
        path = folder / f'{name}.npy'
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        require(sha(path) == recipe()['expected_features'][f'{part}/{name}.npy'], 'Recalculated raw features differ: ' + name)
        files.append(path)
    dump(folder / 'row_ids.json', ids)
    files.append(folder / 'row_ids.json')
    dump(folder / 'manifest.json', input_manifest(folder, files, rows=len(ids), source_sha256=sha(source), label_columns_loaded=False))
    print('PASS: exact raw feature arrays for ' + part)


def preprocessing():
    import joblib
    check_environment()
    ids, _, raw = raw_blocks('train_val')
    names = read(ROOT / 'manifests/c03/feature_schema.json')['rdkit2d']['feature_names']
    require(not (PROCESS / 'processors').exists(), 'Processor output exists')
    for seed in SEEDS:
        frame = split_frame(seed, 'train')
        index = frame.source_row_index.to_numpy(dtype=int)
        require([ids[i] for i in index] == frame.row_id.tolist(), 'Fit membership mismatch')
        processor, available, selected = fit_processor(raw[index])
        output_names = [names[i] for i in selected]
        expected = next(j for j in recipe()['historical_final'] if j['experiment'] == 'S32_CATBOOST' and j['split_seed'] == seed)
        require(output_names == expected['descriptor_names'], 'Historical retained descriptor order differs')
        folder = PROCESS / f'processors/seed_{seed}'
        folder.mkdir(parents=True)
        joblib.dump(processor, folder / 'processor.joblib')
        dump(folder / 'state.json', dict(available_indices=available.tolist(), selected_indices=selected.tolist(),
             output_feature_names=output_names, fit_rows=len(frame), fit_row_ids_sha256=ordered_hash(frame.row_id.tolist())))
        dump(folder / 'manifest.json', input_manifest(folder, [folder / 'state.json', folder / 'processor.joblib'],
             fitted_scope='exact seed-specific official train rows only', test_access=False))
    print('PASS: five newly fitted, train-only processors; no historical serialized processors used')


def audit_structure():
    """Report exact structural overlaps without dropping rows or interpreting labels."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    check_environment()
    source_rows = {}
    for part in ('train_val','test'):
        source = DATA / f'raw/{part}.csv'
        require(sha(source) == recipe()['expected_raw'][part]['sha256'], 'Audit source changed')
        with source.open(newline='') as stream:
            reader=csv.reader(stream)
            header=next(reader)
            smiles=[row[header.index('Drug')] for row in reader]
        mapped=[]
        for i,text in enumerate(smiles):
            mol=Chem.MolFromSmiles(text)
            require(mol is not None and mol.GetNumAtoms()>0,'Invalid molecule in structural audit')
            mapped.append(dict(row_id=f'solubility_aqsoldb:{part}:{i}',raw_smiles=text,
                canonical_smiles=Chem.MolToSmiles(mol,isomericSmiles=True),
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)))
        source_rows[part]=mapped
    rows=[]
    for seed in SEEDS:
        parts={'test':source_rows['test']}
        for role in ('train','valid'):
            frame=split_frame(seed,role)
            parts[role]=[source_rows['train_val'][i] for i in frame.source_row_index]
            rows.append(dict(seed=seed,comparison=role,representation='empty scaffold',overlap_count=sum(r['scaffold']=='' for r in parts[role])))
        for a,b in [('train','valid'),('train','test'),('valid','test')]:
            for key in ['row_id','raw_smiles','canonical_smiles','scaffold']:
                overlap={r[key] for r in parts[a]} & {r[key] for r in parts[b]}
                rows.append(dict(seed=seed,comparison=a+' vs '+b,representation=key,overlap_count=len(overlap)))
    folder=PROCESS/'structure_audit'
    require(not folder.exists(),'Structural audit already exists')
    write_csv(folder/'overlap_counts.csv',rows)
    dump(folder/'manifest.json',input_manifest(folder,[folder/'overlap_counts.csv'],
        test_labels_loaded=False,rows_removed=0,additional_structure_standardization=False,
        interpretation='Finite overlap audit; it does not prove absence of every leakage mechanism'))
    print('Structural overlap counts recorded; no data or model changed')
