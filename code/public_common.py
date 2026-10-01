"""Portable IO, feature processing and experiment boundaries for the public recipe."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/solubility_trackA'
PROCESS = DATA / 'process'
OUTPUT = ROOT / 'generated'
SEEDS = (1, 2, 3, 4, 5)
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(OUTPUT / '.cache/xdg'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def recipe():
    return read(ROOT / 'configs/recipe.json')


def safe_run(name):
    require(name and Path(name).name == name and name not in ('.', '..'), 'Run must be one directory name')
    return OUTPUT / name


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(rows or fields, 'Empty CSV requires explicit columns')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def input_manifest(folder, files, **extra):
    return dict(files_sha256={str(p.relative_to(ROOT)): sha(p) for p in files}, **extra)


def verify(manifest):
    for relative, expected in manifest['files_sha256'].items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT) and path.is_file(), 'Missing/unsafe file: ' + relative)
        require(sha(path) == expected, 'Input changed: ' + relative)


def environment(graph=False):
    import platform
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'catboost', 'rdkit', 'PyTDC']
    if graph:
        names += ['torch', 'chemprop', 'lightning']
    return dict(python=platform.python_version(), packages={name: metadata.version(name) for name in names})


def check_environment(graph=False):
    expected = read(ROOT / 'configs/environment_record.json')
    actual = environment(graph)
    require(actual['python'] == expected['python'], 'Python version differs from recorded environment')
    for name, version in actual['packages'].items():
        require(version == expected['packages'][name], 'Recorded dependency differs: ' + name)
    if graph:
        distribution = metadata.distribution('chemprop')
        for relative, digest in read(ROOT / 'configs/graph_api.json')['installed_source_sha256'].items():
            require(sha(distribution.locate_file(relative)) == digest, 'Recorded native Chemprop source differs: ' + relative)
    return actual


def source_binding():
    files = sorted((ROOT / 'code').glob('public_*.py')) + [ROOT / 'code/reproduce.py', ROOT / 'code/c05_reproduction.py', ROOT / 'manifests/c03/feature_schema.json']
    files += sorted((ROOT / 'configs').glob('*.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def development_guard():
    """Deny held-out inputs during preprocessing fits, selection and model training."""
    def protect(event, args):
        if event == 'socket.connect':
            raise PermissionError('Training and selection are offline')
        if event != 'open' or isinstance(args[0], int):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except TypeError:
            return
        if path.is_relative_to(DATA.resolve()):
            rel = path.relative_to(DATA.resolve())
            require('test' not in rel.parts and path.name != 'test.csv', 'Held-out data are forbidden in development: ' + str(rel))
    sys.addaudithook(protect)


def split_frame(seed, part):
    import pandas as pd
    manifest = read(PROCESS / 'splits/manifest.json')
    record = manifest['splits'][str(seed)][part]
    path = PROCESS / f'splits/seed_{seed}/{part}.csv'
    require(sha(path) == record['csv_sha256'], 'Split CSV changed')
    frame = pd.read_csv(path)
    require(len(frame) == record['rows'] and ordered_hash(frame.row_id.tolist()) == record['row_id_sha256'], 'Split identity changed')
    require(frame.source_partition.eq('train_val').all() and frame.row_id.is_unique, 'Invalid development partition')
    return frame


def raw_blocks(part):
    import numpy as np
    folder = PROCESS / 'features' / part
    manifest = read(folder / 'manifest.json')
    verify(manifest)
    ids = read(folder / 'row_ids.json')
    bits = np.load(folder / 'morgan.npy', allow_pickle=False)
    desc = np.load(folder / 'rdkit2d.npy', allow_pickle=False)
    require(bits.shape == (len(ids), 2048) and bits.dtype == np.uint8 and np.isin(bits, [0, 1]).all(), 'Invalid Morgan array')
    require(desc.shape == (len(ids), 210) and not np.isinf(desc).any(), 'Invalid descriptor array')
    return ids, bits, desc


def fit_processor(matrix):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    available = np.flatnonzero(np.isfinite(matrix).any(axis=0))
    require(len(available) > 0 and not np.isinf(matrix).any(), 'No usable training descriptors')
    processor = Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('variance', VarianceThreshold(threshold=0.0)),
                          ('scaler', StandardScaler())])
    processor.fit(matrix[:, available])
    selected = available[processor.named_steps['variance'].get_support()]
    return processor, available, selected


def load_processor(seed):
    import joblib
    folder = PROCESS / f'processors/seed_{seed}'
    manifest = read(folder / 'manifest.json')
    verify(manifest)
    state = read(folder / 'state.json')
    require(state['fit_row_ids_sha256'] == ordered_hash(split_frame(seed, 'train').row_id.tolist()), 'Processor fitted on different rows')
    # Only locally generated, manifest-verified processor files are deserialized.
    return joblib.load(folder / 'processor.joblib'), state


def view(spec, part, positions=None):
    import numpy as np
    ids, bits, desc = raw_blocks(part)
    index = np.arange(len(ids)) if positions is None else np.asarray(positions, dtype=int)
    require(index.ndim == 1 and len(index) > 0 and np.all((index >= 0) & (index < len(ids))), 'Invalid feature positions')
    representation = spec['representation']
    if representation == 'none':
        return np.zeros((len(index), 1)), []
    blocks, names = [], []
    if representation in ('Morgan', 'Hybrid'):
        blocks.append(bits[index].astype(np.float64))
        names += [f'morgan_bit_{i:04d}' for i in range(2048)]
    if representation in ('RDKit2D', 'Hybrid'):
        processor, state = load_processor(spec['split_seed'])
        transform = processor if spec['algorithm'] == 'ridge' else processor[:-1]
        blocks.append(transform.transform(desc[index][:, state['available_indices']]))
        names += state['output_feature_names']
    x = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    if spec['experiment'] in ('S21_RF', 'S22_CATBOOST', 'S31_RF', 'S32_CATBOOST'):
        column = names.index('Ipc')
        require(np.isfinite(x).all() and (x[:, column] >= 0).all(), 'Invalid Ipc input')
        x = x.copy()
        x[:, column] = np.log1p(x[:, column])
    require(np.isfinite(x).all(), 'Nonfinite model inputs')
    if spec['algorithm'] in ('random_forest', 'catboost'):
        require(np.isfinite(x.astype(np.float32)).all(), 'Tree feature float32 overflow')
    return x, names


def matrix_sha(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def stable_sd(values, ddof=1, axis=None):
    """Retain finite extreme Ridge results without squaring their original scale."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    safe = np.where(scale == 0, 1, scale)
    return np.std(values / safe, axis=axis, ddof=ddof) * np.squeeze(safe, axis=axis)


def metrics(y, p):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    require(y.shape == p.shape and y.ndim == 1 and np.isfinite(y).all() and np.isfinite(p).all(), 'Invalid metric coverage')
    constant = np.ptp(y) == 0 or np.ptp(p) == 0
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)), Spearman=None if constant else float(spearmanr(y, p).statistic))
