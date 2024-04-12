import os, glob

import polars as pl
import numpy as np
import pandas as pd

import predata

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
    
    def cat_to_int_encode(self, data: pl.DataFrame, column_name: str, new_column_name: str, encoding_dict: dict) -> pl.DataFrame:
        """
        Convert a categorical column to integer inplace.
        """
        # create list of dict keys:
        roles_ordered = list(encoding_dict.keys())

        # Check if all unique values are in roles_ordered, if not use replace method
        unique_values = data[column_name].drop_nulls().unique().to_list()
        if all(val in roles_ordered for val in unique_values):
            try:   # Enum encoding
                data = data.with_columns(pl.col(column_name).cast(pl.Enum(roles_ordered)).to_physical().alias(new_column_name))
            except:
                print(f"Error with {column_name} encoding using Enum. Switching to replace ...")
                data = data.with_columns(pl.col(column_name).replace(encoding_dict, default=None).alias(new_column_name))
        else:
            print("Not all unique values are in roles_ordered. Switching to replace ...")
            data = data.with_columns(pl.col(column_name).replace(encoding_dict, default=None).alias(new_column_name))
        
        return data
    
    def encode_categorical_columns(self, data: pl.DataFrame, table_name: str) -> pl.DataFrame:
        """
        Encode categorical columns:
        - data: DataFrame to encode
        - return: encoded DataFrame
        """

        def norm_frequency(frequency_dict: dict) -> dict:
            """
            Normalize frequency dictionary by dividing each value by the sum of all values.
            """
            total_sum = sum(frequency_dict.values())
            return {key: value / total_sum for key, value in frequency_dict.items()}


        if table_name=='person_2':

            #################################
            ###  relatedpersons_role_762T ###
            #################################
            # Cast to Enum, otherwise use replace method
            # roles_encoding = {"OTHER": 0, "COLLEAGUE": 1, "FRIEND": 2, "NEIGHBOR": 3, "OTHER_RELATIVE": 4, "CHILD": 5, "SIBLING": 6, "GRAND_PARENT": 7, "PARENT": 8, "SPOUSE": 9}
            # Load predefined dictionary
            roles_encoding = predata.relatedpersons_role_762T_encoding
            data = self.cat_to_int_encode(data, "relatedpersons_role_762T", "relatedpersons_role_encoded", roles_encoding)

            #################################
            ###    addres_district_368M   ###
            #################################
            # Chain an addition of several new columns
            # TODO: mormalize frequency by norm_frequency(predata.relatedpersons_role_762T_mean_target)
            # TODO: add another fill_null('None') in the end of chain?
            data = data.with_columns(
                pl.col("relatedpersons_role_762T").fill_null('None').replace(predata.relatedpersons_role_762T_mean_target, default=None).alias("relatedpersons_role_mean_target"),
                pl.col("relatedpersons_role_762T").fill_null('None').replace(predata.relatedpersons_role_762T_frequency, default=None).alias("relatedpersons_role_frequency"),

                pl.col("addres_district_368M").replace(predata.addres_district_368M_mean_target, default=None).alias("addres_district_mean_target"),
                pl.col("addres_district_368M").replace(predata.addres_district_368M_frequency, default=None).alias("addres_district_frequency"),

                pl.col("addres_role_871L").fill_null('None').replace(predata.addres_role_871L_mean_target, default=None).alias("addres_role_mean_target"),
                pl.col("addres_role_871L").fill_null('None').replace(predata.addres_role_871L_frequency, default=None).alias("addres_role_frequency"),
            ).drop(["addres_district_368M", "relatedpersons_role_762T", "addres_role_871L"])
            
            # Convert dictionary elements from counts to frequency by normalizing each element by the sum of elements
            relatedpersons_role_762T_to_frequency = predata.relatedpersons_role_762T_to_frequency
            for key in relatedpersons_role_762T_to_frequency:
                relatedpersons_role_762T_to_frequency[key] /= sum(relatedpersons_role_762T_to_frequency.values())

        return data        
    
    def aggregate_depth_2(self, data: pl.DataFrame, table_name: str, smart_features: bool):
        """
        Aggregate the data by depth 2:
        - data: DataFrame to aggregate
        - table_name: name of the table to aggregate
        - smart_features: flag to enable smart features
        - return: aggregated DataFrame
        """

        # Create a new DataFrame to store the aggregated results
        #result = pl.DataFrame()

        if table_name=='person_2' and smart_features:
            # Encode categorical columns
            data = self.encode_categorical_columns(data, table_name)

            
            data = data.group_by(['case_id','num_group1']).agg(
                    # Number of non-null related person roles indicated
                    pl.when(pl.col("relatedpersons_role_762T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_related_persons"),
                    # The most influential role
                    pl.col("relatedpersons_role_encoded_762T").max().alias("most_influential_role"),
                )
            
        elif table_name=='person_2' and not smart_features:
            # Create columns with 0/1 for each role in relatedpersons_role_762T
            roles_ordered = ["OTHER", "COLLEAGUE", "FRIEND", "NEIGHBOR", "OTHER_RELATIVE", "CHILD", "SIBLING", "GRAND_PARENT", "PARENT", "SPOUSE"]
            for role in roles_ordered:
                data = data.with_columns(
                    pl.col("relatedpersons_role_762T").eq(role).cast(pl.Int8).alias(f"relatedpersons_role_{role}_762T")
                )

            # Dropping unneeded columns
            data = data.drop(["relatedpersons_role_762T"])

            # Aggregating by case_id and num_group1
            data = data.group_by(['case_id', 'num_group1']).agg(
                # Agrregate all "relatedpersons_role_{role}_762T" columns for each role in roles_ordered as a sum
                *[pl.col(f"relatedpersons_role_{role}_762T").sum().cast(pl.Int8).alias(f"num_related_persons_{role}") for role in roles_ordered],
            )



        return data

    def add_target(self, data: pl.DataFrame, train_basetable: pl.DataFrame):
        """
        Add the target column to the DataFrame:
        - data: DataFrame to add the target column to
        - train_basetable: DataFrame to get the target column from
        - return: DataFrame with the target column added
        """
        data = data.join(
            train_basetable.select(['case_id', 'WEEK_NUM' , 'target']), on='case_id', how='left'
        )
        
        return data

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
