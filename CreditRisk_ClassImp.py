import os, glob

import polars as pl
import numpy as np
import pandas as pd

dataPath = "/kaggle/input/home-credit-credit-risk-model-stability/"



class DataLoader:
    """
    Load data from parquet files.
    """
    def __init__(self, data_path: str, data_type: str = 'train', complete: bool = False):
        self.data_path = data_path
        self.data_type = data_type
        self.complete = complete

    def load_data(self):

        data_list = ['base', 'static_0', 'static_cb_0', 'applprev_1', 'other_1', 'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1', 'credit_bureau_a_1',
                     'credit_bureau_b_1', 'deposit_1', 'person_1', 'debitcard_1', 'applprev_2', 'person_2', 'credit_bureau_a_2', 'credit_bureau_b_2']
        if not self.complete:
            data_list = data_list[:4]

        data_store = {}
        for data in data_list:
            data_store[data] = glob.glob('{data_path}parquet_files/{data_type}/{data_type}_{data}*.parquet'.format(data_path=self.data_path, data_type=self.data_type, data=data))

        # data_store = {
        #     'base': ['{data_path}parquet_files/{data_type}/{data_type}_base.parquet'.format(data_path=self.data_path, data_type=self.data_type)],
        #     'static_0': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_static_0*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'static_cb_0': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_static_cb_0*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'applprev_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_applprev_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'other_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_other_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'tax_registry_a_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_tax_registry_a_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'tax_registry_b_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_tax_registry_b_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'tax_registry_c_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_tax_registry_c_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'credit_bureau_a_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_credit_bureau_a_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'credit_bureau_b_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_credit_bureau_b_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'deposit_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_deposit_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'person_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_person_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'debitcard_1': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_debitcard_1*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'applprev_2': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_applprev_2*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'person_2': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_person_2*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'credit_bureau_a_2': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_credit_bureau_a_2*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        #     'credit_bureau_b_2': glob.glob('{data_path}parquet_files/{data_type}/{data_type}_credit_bureau_b_2*.parquet'.format(data_path=self.data_path, data_type=self.data_type)),
        # }



        data_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        data_dfs = []
        for data_file in data_files:
            data_df = pl.read_csv(data_file)
            data_dfs.append(data_df)
        data_df = pl.concat(data_dfs)
        return data_df
