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

    def set_table_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cast the DataFrame to the desired dtypes.
        """

        # To-Do: check more efficient typings and add protections (cast give error if out of range)
        data_type_mappings = {
            'L': pl.Int64,
            'A': pl.Float64,
            'D': pl.Date,
            'M': pl.String,
            'T': pl.String,
            'P': pl.Float64
        }

        # implement here all desired dtypes for tables
        for col in df.columns:
            # casting specific columns
            if col in ['case_id', 'WEEK_NUM', 'num_group1', 'num_group2']:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
                continue # Move to the next column
            else:
                # last letter of column name will help you determine the type
                for ending, data_type in data_type_mappings.items():
                    if col[-1] == ending:
                        df = df.with_columns(pl.col(col).cast(data_type))
                        break # Move to the next column

        return df

    def aggregate(self, df: pl.DataFrame, group_by: list, agg_dict: dict) -> pl.DataFrame:
        """
        Group the data by group_by and summarize the data by aggregated values:
        - group_by: list of columns to group by
        - agg_dict: dictionary of columns to aggregate and their aggregation functions
        - return: aggregated DataFrame
        """
        aggregation_rules_by_endings = {
            'L': [pl.max, pl.min, pl.first, pl.last],
            'A': [pl.max, pl.min, pl.first, pl.last, pl.mean],
            'D': [pl.max, pl.min, pl.first, pl.last, pl.mean],
            'M': [pl.first, pl.last, pl.mode],
            'T': [pl.first, pl.last],
            'P': [pl.max, pl.min, pl.first, pl.last, pl.mean],
        }

        df_columns = df.columns
        for col in df_columns:
            if col[-1] in aggregation_rules_by_endings:
                for agg_func in aggregation_rules_by_endings[col[-1]]:
                    agg_dict[col] = agg_dict.get(col, []) + [agg_func]
                    if col not in group_by:
                        group_by.append(col)
                        break # Move to the next column
                    else:
                        continue # Move to the next column
                    #break # Move to the next column


        #df = df.groupby(group_by).agg(agg_dict)
        return df

    def load_data(self):

        # List of data tables to be loaded
        data_list = ['base', 'static_0', 'static_cb_0',
                     'applprev_1', 'other_1', 'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1', 'credit_bureau_a_1',
                     'credit_bureau_b_1', 'deposit_1', 'person_1', 'debitcard_1',
                     'applprev_2', 'person_2', 'credit_bureau_a_2', 'credit_bureau_b_2']
        if not self.complete:
            data_list = data_list[:4]

        # Relations between data tables for concatination of depth 1 and 2 after aggregation of depth 2
        data_relations_by_depth = {
            'applprev_1': 'applprev_2',
            'person_1': 'person_2',
            'credit_bureau_a_1': 'credit_bureau_a_2',
            'credit_bureau_b_1': 'credit_bureau_b_2'
        }

        data_store = {}
        for data in data_list:
            data_files = glob.glob('{data_path}parquet_files/{data_type}/{data_type}_{data}*.parquet'.format(data_path=self.data_path, data_type=self.data_type, data=data))

            if len(data_files) == 0:
                raise Exception(f"No parquet files found for {data}")
            elif len(data_files) == 1:
                data_store[data] = pl.read_parquet(data_files[0].pipe(self.set_table_dtypes))
            else:
                data_store[data] = pl.concat([pl.read_parquet(file).pipe(self.set_table_dtypes) for file in data_files], how="vertical_relaxed")

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



        return data_store
