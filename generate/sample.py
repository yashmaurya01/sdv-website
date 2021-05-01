import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
sys.setrecursionlimit(3000)

from .tablegan_fixed import TableGAN
from sdgym import benchmark
import pandas as pd
from sdgym.datasets import get_dataset_paths, load_dataset, load_tables
from sdgym.synthesizers import (
    CLBN, CopulaGAN, CTGAN)
from sdgym.utils import (
    build_synthesizer, format_exception, get_synthesizers_dict, import_object, used_memory)
from sdgym.synthesizers.utils import TableganTransformer
import numpy as np
import rdt
import sdv

def sample_tablegan(dataset_name, table_name, dataset_root_folder, output=None, sample_synthetic_rows=10000, tablegan_optional_parameters={}, preprocess_table=lambda x: x):
    datasets = get_dataset_paths([dataset_name], dataset_root_folder, "")
    print(datasets[0].resolve())
    metadata = load_dataset(datasets[0])
    real_data = load_tables(metadata)
    print(metadata)
    print(metadata.modality)
    print("ACCEPTABLE MODULALITIES", get_synthesizers_dict(TableGAN)[0].get('modalities'))
    print(real_data.items())
    mm = metadata.get_table_meta(table_name)
    data = real_data[table_name]
    data = preprocess_table(data)
    real_data[table_name] = data
    gc = TableGAN(**tablegan_optional_parameters)
    columns, categoricals = gc._get_columns(data, mm)
    gc.update_column_info(columns)
    ht = rdt.HyperTransformer(dtype_transformers={
        'O': 'label_encoding',
    })
    ht.fit(data.iloc[:, categoricals])
    gc.fit_sample(real_data.copy(), metadata)
    s = gc.sample(sample_synthetic_rows)
    s = pd.DataFrame(s, columns=columns)
    vv = data[0:0]
    vv = vv.append(ht.reverse_transform(s), ignore_index=True)
    if output is None:
        output = dataset_name + "_synthetic.csv"
    vv.to_csv(output)
    return vv


def sample_ctgan(dataset_name, table_name, dataset_root_folder, output=None, sample_synthetic_rows=10000, preprocess_table=lambda x: x):
    datasets = get_dataset_paths([dataset_name], dataset_root_folder, "")
    print(datasets[0].resolve())
    metadata = load_dataset(datasets[0])
    real_data = load_tables(metadata)
    data = real_data[table_name]
    data = preprocess_table(data)
    real_data[table_name] = data
    print(metadata)
    gc = TableGAN()
    mm = metadata.get_table_meta(table_name)
    columns, categoricals = gc._get_columns(data, mm)
    gc = sdv.tabular.CTGAN(table_metadata=mm)
    gc.fit(real_data[table_name])
    s = gc.sample(sample_synthetic_rows)
    s = pd.DataFrame(s, columns=columns)
    vv = data[0:0]
    vv = vv.append(s, ignore_index=True)
    if output is None:
        output = dataset_name + "_synthetic.csv"
    vv.to_csv(output)
    return vv