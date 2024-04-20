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

                pl.col("addres_zip_823M").replace(predata.addres_zip_823M_mean_target, default=None).alias("addres_zip_mean_target"),
                pl.col("addres_zip_823M").replace(predata.addres_zip_823M_frequency, default=None).alias("addres_zip_frequency"),

                pl.col("conts_role_79M").replace(predata.conts_role_79M_mean_target, default=None).alias("conts_role_mean_target"),
                pl.col("conts_role_79M").replace(predata.conts_role_79M_frequency, default=None).alias("conts_role_frequency"),

                pl.col("empls_economicalst_849M").replace(predata.empls_economicalst_849M_mean_target, default=None).alias("empls_economicalst_mean_target"),
                pl.col("empls_economicalst_849M").replace(predata.empls_economicalst_849M_frequency, default=None).alias("empls_economicalst_frequency"),

            )#.drop(["addres_district_368M", "relatedpersons_role_762T", "addres_role_871L", "addres_zip_823M", "conts_role_79M", "empls_economicalst_849M",
             # "empls_employer_name_740M"])

             # Dropped completely: empls_employer_name_740M, 
        
        if table_name=='applprev_2':
            contact_type_encoding = predata.conts_type_509L_encoding
            data = self.cat_to_int_encode(data, "conts_type_509L", "conts_type_encoded", contact_type_encoding)

            credacc_cards_status_encoding = predata.credacc_cards_status_52L_encoding
            data = self.cat_to_int_encode(data, "credacc_cards_status_52L", "credacc_cards_status_encoded", credacc_cards_status_encoding)

            cacccardblochreas_147M_encoding = predata.cacccardblochreas_147M_encoding
            data = self.cat_to_int_encode(data, "cacccardblochreas_147M", "cacccardblochreas_encoded", cacccardblochreas_147M_encoding)

            # Adding new columns
            data = data.with_columns(
                pl.col("conts_type_509L").fill_null('None').replace(predata.conts_type_509L_mean_target, default=None).alias("conts_type_mean_target"),
                pl.col("conts_type_509L").fill_null('None').replace(predata.conts_type_509L_frequency, default=None).alias("conts_type_frequency"),

                pl.col("credacc_cards_status_52L").fill_null('None').replace(predata.credacc_cards_status_52L_mean_target, default=None).alias("credacc_cards_status_mean_target"),
                pl.col("credacc_cards_status_52L").fill_null('None').replace(predata.credacc_cards_status_52L_frequency, default=None).alias("credacc_cards_status_frequency"),

                pl.col("cacccardblochreas_147M").fill_null('None').replace(predata.cacccardblochreas_147M_mean_target, default=None).alias("cacccardblochreas_147M_mean_target"),
                pl.col("cacccardblochreas_147M").fill_null('None').replace(predata.cacccardblochreas_147M_frequency, default=None).alias("cacccardblochreas_147M_frequency"),
            )

            drop_columns = ["conts_type_509L", "credacc_cards_status_52L", "cacccardblochreas_147M"]

        if table_name=='credit_bureau_a_2':

            collater_typofvalofguarant_unique = ['9a0c095e','8fd95e4b','06fb9ba8','3cbe86ba']

            # Adding new columns
            data = data.with_columns(
                pl.col("collater_typofvalofguarant_298M").replace(predata.collater_typofvalofguarant_298M_mean_target, default=None).alias("collater_typofvalofguarant_298M_mean_target"),
                pl.col("collater_typofvalofguarant_298M").replace(predata.collater_typofvalofguarant_298M_frequency, default=None).alias("collater_typofvalofguarant_298M_frequency"),

                # Add columns as one-hot-encoded values of collater_typofvalofguarant_298M
                *[pl.col("collater_typofvalofguarant_298M").eq(role).cast(pl.Int8).alias(f"collater_typofvalofguarant_298M_{role}") for role in collater_typofvalofguarant_unique],

                pl.col("collater_typofvalofguarant_407M").replace(predata.collater_typofvalofguarant_407M_mean_target, default=None).alias("collater_typofvalofguarant_407M_mean_target"),
                pl.col("collater_typofvalofguarant_407M").replace(predata.collater_typofvalofguarant_407M_frequency, default=None).alias("collater_typofvalofguarant_407M_frequency"),

                # Add columns as one-hot-encoded values of collater_typofvalofguarant_407M
                *[pl.col("collater_typofvalofguarant_407M").eq(role).cast(pl.Int8).alias(f"collater_typofvalofguarant_407M_{role}") for role in collater_typofvalofguarant_unique],

                pl.col("collaterals_typeofguarante_359M").replace(predata.collaterals_typeofguarante_359M_mean_target, default=None).alias("collaterals_typeofguarante_359M_mean_target"),
                pl.col("collaterals_typeofguarante_359M").replace(predata.collaterals_typeofguarante_359M_frequency, default=None).alias("collaterals_typeofguarante_359M_frequency"),

                pl.col("collaterals_typeofguarante_669M").replace(predata.collaterals_typeofguarante_669M_mean_target, default=None).alias("collaterals_typeofguarante_669M_mean_target"),
                pl.col("collaterals_typeofguarante_669M").replace(predata.collaterals_typeofguarante_669M_frequency, default=None).alias("collaterals_typeofguarante_669M_frequency"),

                pl.col("subjectroles_name_541M").replace(predata.subjectroles_name_541M_mean_target, default=None).alias("subjectroles_name_541M_mean_target"),
                pl.col("subjectroles_name_541M").replace(predata.subjectroles_name_541M_frequency, default=None).alias("subjectroles_name_541M_frequency"),

                pl.col("subjectroles_name_838M").replace(predata.subjectroles_name_838M_mean_target, default=None).alias("subjectroles_name_838M_mean_target"),
                pl.col("subjectroles_name_838M").replace(predata.subjectroles_name_838M_frequency, default=None).alias("subjectroles_name_838M_frequency"),
            )

            drop_columns = ["collater_typofvalofguarant_298M", "collater_typofvalofguarant_407M", "collaterals_typeofguarante_359M", "collaterals_typeofguarante_669M"]

        if table_name=='credit_bureau_b_2':

            data = data.with_columns(
                # Fill null with 0 for pmts_dpdvalue_108P column
                pl.col("pmts_dpdvalue_108P").fill_null(0).alias("pmts_dpdvalue_108P"),
            )

            def count_intervals(df):
                # Assuming 'df' is sorted by date or some time order
                return df.with_column(
                    pl.col("pmts_dpdvalue_108P").is_not_null()
                    .and_(pl.col("pmts_dpdvalue_108P").gt(0))
                    .and_(pl.col("pmts_dpdvalue_108P").shift(-1).is_not_null())
                    .and_(pl.col("pmts_dpdvalue_108P").shift(-1).gt(0))
                    .and_(pl.col("pmts_dpdvalue_108P").shift(-1).ne(pl.col("pmts_dpdvalue_108P")))
                    .cumsum()
                    .alias("intervals")
                    )


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
                    pl.col("relatedpersons_role_encoded").max().alias("most_influential_role"),
                    # Start date of employment
                    pl.col("empls_employedfrom_796D").first().alias('empls_employedfrom_796D'),
                    # Various mean_target columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                    # Various frequency columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                )
            
        elif table_name=='applprev_2' and smart_features:
            # Encode categorical columns
            data = self.encode_categorical_columns(data, table_name)

            data = data.group_by(['case_id','num_group1']).agg(
                    # Number of non-null contact roles indicated
                    pl.when(pl.col("conts_type_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_contacts"),
                    # The most influential contact
                    pl.col("conts_type_encoded").max().alias("most_influential_contact"),

                    # Number of non-null credacc_cards_status
                    # TODO: check .replace("a55475b1", None) addition
                    pl.when(pl.col("credacc_cards_status_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_credacc_cards_status"),
                    # The most influential credacc_cards_status
                    pl.col("credacc_cards_status_encoded").max().alias("most_influential_credacc_cards_status"),

                    # Number of credit card blocks
                    # TODO: check .replace("a55475b1", None) addition
                    pl.when(pl.col("cacccardblochreas_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_credit_card_blocks"),
                    # The most influential credit card block
                    pl.col("cacccardblochreas_encoded").max().alias("most_influential_credit_card_block"),

                    # Various mean_target columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                    # Various frequency columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                )
            
        elif table_name=='credit_bureau_a_2' and smart_features:
            # Encode categorical columns
            data = self.encode_categorical_columns(data, table_name)

            collater_typofvalofguarant_unique = ['9a0c095e','8fd95e4b','06fb9ba8','3cbe86ba']

            data = data.group_by(['case_id', 'num_group1']).agg(
                    # Number of non-null collater_typofvalofguarant_298M
                    pl.when(pl.col("collater_typofvalofguarant_298M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collater_typofvalofguarant_298M"),
                    # Sum of one-hot-encoded columns
                    *[pl.col(f"collater_typofvalofguarant_298M_{role}").sum().cast(pl.Int16).alias(f"collater_typofvalofguarant_{role}_298M") for role in collater_typofvalofguarant_unique],

                    # Number of non-null collater_typofvalofguarant_407M
                    pl.when(pl.col("collater_typofvalofguarant_407M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collater_typofvalofguarant_407M"),
                    # Sum of one-hot-encoded columns
                    *[pl.col(f"collater_typofvalofguarant_407M_{role}").sum().cast(pl.Int16).alias(f"collater_typofvalofguarant_{role}_407M") for role in collater_typofvalofguarant_unique],

                    # Number of non-null collater_valueofguarantee_1124L
                    pl.when(pl.col("collater_valueofguarantee_1124L").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collater_valueofguarantee_1124L"),
                    # Total sum and mean of collater_valueofguarantee_1124L
                    pl.col("collater_valueofguarantee_1124L").sum().alias("collater_valueofguarantee_1124L_sum"),
                    pl.col("collater_valueofguarantee_1124L").mean().alias("collater_valueofguarantee_1124L_mean"),

                    # Number of non-null collater_valueofguarantee_876L
                    pl.when(pl.col("collater_valueofguarantee_876L").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collater_valueofguarantee_876L"),
                    # Total sum and mean of collater_valueofguarantee_876L
                    pl.col("collater_valueofguarantee_876L").sum().alias("collater_valueofguarantee_876L_sum"),
                    pl.col("collater_valueofguarantee_876L").mean().alias("collater_valueofguarantee_876L_mean"),

                    # Number of non-null collaterals_typeofguarante_359M
                    pl.when(pl.col("collaterals_typeofguarante_359M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collaterals_typeofguarante_359M"),
                    # Number of non-null collaterals_typeofguarante_669M
                    pl.when(pl.col("collaterals_typeofguarante_669M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_collaterals_typeofguarante_669M"),

                    # Days past due of the payment columns (pmts_dpd_1073P)
                    pl.when(
                            (pl.col("pmts_dpd_1073P").is_not_null()) & (pl.col("pmts_dpd_1073P").gt(0.0))
                        ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_dpd_1073P"),
                    pl.col("pmts_dpd_1073P").sum().alias("pmts_dpd_1073P_sum"),
                    pl.col("pmts_dpd_1073P").mean().alias("pmts_dpd_1073P_mean"),

                    # Days past due of the payment columns (pmts_dpd_303P)
                    pl.when(
                            (pl.col("pmts_dpd_303P").is_not_null()) & (pl.col("pmts_dpd_303P").gt(0.0))
                        ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_dpd_303P"),
                    pl.col("pmts_dpd_303P").sum().alias("pmts_dpd_303P_sum"),
                    pl.col("pmts_dpd_303P").mean().alias("pmts_dpd_303P_mean"),

                    # Overdue payment
                    pl.when(
                            (pl.col("pmts_overdue_1140A").is_not_null()) & (pl.col("pmts_overdue_1140A").gt(0.0))
                        ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_pmts_overdue_1140A"),
                    pl.col("pmts_overdue_1140A").sum().alias("pmts_overdue_1140A_sum"),
                    pl.col("pmts_overdue_1140A").mean().alias("pmts_overdue_1140A_mean"),
                    pl.col("pmts_overdue_1140A").filter(
                            (pl.col("pmts_overdue_1140A").is_not_null()) & (pl.col("pmts_overdue_1140A").gt(0.0))
                        ).last().alias("pmts_overdue_1140A_last"),

                    pl.when(
                            (pl.col("pmts_overdue_1152A").is_not_null()) & (pl.col("pmts_overdue_1152A").gt(0.0))
                        ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_pmts_overdue_1152A"),
                    pl.col("pmts_overdue_1152A").sum().alias("pmts_overdue_1152A_sum"),
                    pl.col("pmts_overdue_1152A").mean().alias("pmts_overdue_1152A_mean"),
                    pl.col("pmts_overdue_1152A").filter(
                            (pl.col("pmts_overdue_1152A").is_not_null()) & (pl.col("pmts_overdue_1152A").gt(0.0))
                        ).last().alias("pmts_overdue_1152A_last"),

                    # Number of non-null subjectroles_name_541M
                    pl.when(pl.col("subjectroles_name_541M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_subjectroles_name_541M"),

                    # Years of payments of closed credit (pmts_year_507T) contract and the current contract (pmts_year_1139T)
                    # Number of non-null pmts_year_507T
                    pl.when(pl.col("pmts_year_507T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_pmts_year_507T"),
                    # First and last years of payments of closed credit (pmts_year_507T) contract, avoiding null, and their difference
                    pl.col("pmts_year_507T").cast(pl.Float64).min().alias("pmts_year_507T_first"),
                    pl.col("pmts_year_507T").cast(pl.Float64).max().alias("pmts_year_507T_last"),
                    (pl.col("pmts_year_507T").cast(pl.Float64).max() - pl.col("pmts_year_507T").cast(pl.Float64).min()).alias("pmts_year_507T_duration"),
                    # Number of non-null pmts_year_507T
                    pl.when(pl.col("pmts_year_1139T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_pmts_year_1139T"),
                    # First and last years of payments of the current credit (pmts_year_1139T) contract, avoiding null, and their difference
                    pl.col("pmts_year_1139T").cast(pl.Float64).min().alias("pmts_year_1139T_first"),
                    pl.col("pmts_year_1139T").cast(pl.Float64).max().alias("pmts_year_1139T_last"),
                    (pl.col("pmts_year_1139T").cast(pl.Float64).max() - pl.col("pmts_year_1139T").cast(pl.Float64).min()).alias("pmts_year_1139T_duration"),

                    # Number of years without credit
                    (pl.col("pmts_year_1139T").cast(pl.Float64).min() - pl.col("pmts_year_507T").cast(pl.Float64).max()).alias("pmts_year_1139T_507T_diff"),

                    # Various mean_target columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                    # Various frequency columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],

            )

            # Dropped completely: pmts_month_158T, pmts_month_706T
            # Implemented: pmts_year_1139T, pmts_year_507T
            
        elif table_name=='credit_bureau_b_2' and smart_features:
            # Fill null for pmts_dpdvalue_108P (observed)
            data = self.encode_categorical_columns(data, table_name)

            data = data.group_by(['case_id', 'num_group1']).agg(
                # Number of non-null pmts_date_1107D (it has type pl.Date)
                pl.when(pl.col("pmts_date_1107D").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_pmts_date_1107D"),
                # First and last years of payments of the active contract pmts_date_1107D as well as duration
                pl.col("pmts_date_1107D").min().dt.year().alias("pmts_date_1107D_first"),
                pl.col("pmts_date_1107D").max().dt.year().alias("pmts_date_1107D_last"),
                (pl.col("pmts_date_1107D").max().dt.year() - pl.col("pmts_date_1107D").min().dt.year()).alias("pmts_date_1107D_duration"),
                (pl.col("pmts_date_1107D").max() - pl.col("pmts_date_1107D").min()).dt.total_days().alias("pmts_date_1107D_duration_days"),

                # pmts_dpdvalue_108P values (TODO: is this money or days?)
                pl.when(
                        (pl.col("pmts_dpdvalue_108P").is_not_null()) & (pl.col("pmts_dpdvalue_108P").gt(0.0))
                    ).then(1).otherwise(0).sum().cast(pl.Int32).alias("num_pmts_dpdvalue_108P"),
                pl.col("pmts_dpdvalue_108P").max().alias("pmts_dpdvalue_108P_max"),
                pl.col("pmts_dpdvalue_108P").last().alias("pmts_dpdvalue_108P_last"),
                pl.col("pmts_dpdvalue_108P").filter(
                        (pl.col("pmts_dpdvalue_108P").is_not_null()) & (pl.col("pmts_dpdvalue_108P").gt(0.0))
                    ).arg_max().fill_null(-1).alias("pmts_dpdvalue_108P_maxidx"),
                # TODO: check here which positive trend is better
                #(pl.col("pmts_dpdvalue_108P").max() - pl.col("pmts_dpdvalue_108P").last()).alias("pmts_dpdvalue_108P_pos"),
                ((pl.col("pmts_dpdvalue_108P").max() - pl.col("pmts_dpdvalue_108P").last())/pl.col("pmts_dpdvalue_108P").max()).fill_nan(1.0).alias("pmts_dpdvalue_108P_pos"),

                # pmts_pmtsoverdue_635A values
                pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
                pl.col("pmts_pmtsoverdue_635A").last().alias("pmts_pmtsoverdue_635A_last"),
                pl.col("pmts_pmtsoverdue_635A").filter(
                        (pl.col("pmts_pmtsoverdue_635A").is_not_null()) & (pl.col("pmts_pmtsoverdue_635A").gt(0.0))
                    ).arg_max().fill_null(-1).alias("pmts_pmtsoverdue_635A_maxidx"),
                ((pl.col("pmts_pmtsoverdue_635A").max() - pl.col("pmts_pmtsoverdue_635A").last())/pl.col("pmts_pmtsoverdue_635A").max()).fill_nan(1.0).alias("pmts_pmtsoverdue_635A_pos"),
        

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
