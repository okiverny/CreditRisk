import os, glob
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

import predata
import gc


class CreditRiskProcessing:
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
            #'L': pl.String,
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
    
    def reduce_memory_usage_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types."""
    
        print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
        Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
        Numeric_Float_types = [pl.Float32,pl.Float64]
        for col in df.columns:
            if col in ['case_id', 'WEEK_NUM', 'num_group1', 'num_group2']:
                continue

            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            # Ignore columns with all nulls
            if not c_min: continue

            # Casting
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            #elif col_type == pl.Utf8:
            #    df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass

        print(f"Memory usage after optimization is: {round(df.estimated_size('mb'), 2)} MB")
        return df

    def create_ordinal_encoding(self, mean_target_dict):
        # Handle None values by assigning them a high value for sorting
        sorted_categories = sorted(mean_target_dict.items(), key=lambda x: (float('inf') if x[1] is None else x[1]))
        # Create the ordinal encoding dictionary
        return {category: idx for idx, (category, _) in enumerate(sorted_categories)}


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
            #roles_encoding = predata.relatedpersons_role_762T_encoding
            #data = self.cat_to_int_encode(data, "relatedpersons_role_762T", "relatedpersons_role_encoded", roles_encoding)

            # Chain an addition of several new columns
            # TODO: mormalize frequency by norm_frequency(predata.relatedpersons_role_762T_mean_target)
            # TODO: add another fill_null('None') in the end of chain?
            data = data.with_columns(
                # Categorical ordered encoding
                pl.col("relatedpersons_role_762T").fill_null('None').replace(predata.relatedpersons_role_762T_mean_target, default=None).alias("relatedpersons_role_mean_target"),
                pl.col("relatedpersons_role_762T").fill_null('None').replace(predata.relatedpersons_role_762T_frequency, default=None).alias("relatedpersons_role_frequency"),
                pl.col("relatedpersons_role_762T").fill_null('None').replace(self.create_ordinal_encoding(predata.relatedpersons_role_762T_mean_target), default=None).alias("relatedpersons_role_762T_encoded"),

                pl.col("addres_district_368M").replace(predata.addres_district_368M_mean_target, default=None).alias("addres_district_mean_target"),
                pl.col("addres_district_368M").replace(predata.addres_district_368M_frequency, default=None).alias("addres_district_frequency"),
                pl.col("addres_district_368M").replace(self.create_ordinal_encoding(predata.addres_district_368M_mean_target), default=None).alias("addres_district_368M_encoded"),

                pl.col("addres_role_871L").cast(pl.String).fill_null('None').replace(predata.addres_role_871L_mean_target, default=None).alias("addres_role_mean_target"),
                pl.col("addres_role_871L").cast(pl.String).fill_null('None').replace(predata.addres_role_871L_frequency, default=None).alias("addres_role_frequency"),
                pl.col("addres_role_871L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.addres_role_871L_mean_target), default=None).alias("addres_role_871L_encoded"),

                pl.col("addres_zip_823M").replace(predata.addres_zip_823M_mean_target, default=None).alias("addres_zip_mean_target"),
                pl.col("addres_zip_823M").replace(predata.addres_zip_823M_frequency, default=None).alias("addres_zip_frequency"),
                pl.col("addres_zip_823M").replace(self.create_ordinal_encoding(predata.addres_zip_823M_mean_target), default=None).alias("addres_zip_823M_encoded"),

                pl.col("conts_role_79M").replace(predata.conts_role_79M_mean_target, default=None).alias("conts_role_mean_target"),
                pl.col("conts_role_79M").replace(predata.conts_role_79M_frequency, default=None).alias("conts_role_frequency"),
                pl.col("conts_role_79M").replace(self.create_ordinal_encoding(predata.conts_role_79M_mean_target), default=None).alias("conts_role_79M_encoded"),

                pl.col("empls_economicalst_849M").replace(predata.empls_economicalst_849M_mean_target, default=None).alias("empls_economicalst_mean_target"),
                pl.col("empls_economicalst_849M").replace(predata.empls_economicalst_849M_frequency, default=None).alias("empls_economicalst_frequency"),
                pl.col("empls_economicalst_849M").replace(self.create_ordinal_encoding(predata.empls_economicalst_849M_mean_target), default=None).alias("empls_economicalst_849M_encoded"),

            )#.drop(["addres_district_368M", "relatedpersons_role_762T", "addres_role_871L", "addres_zip_823M", "conts_role_79M", "empls_economicalst_849M",
             # "empls_employer_name_740M"])

             # Dropped completely: empls_employer_name_740M, 
        
        if table_name=='applprev_2':
            # contact_type_encoding = predata.conts_type_509L_encoding
            # data = self.cat_to_int_encode(data, "conts_type_509L", "conts_type_encoded", contact_type_encoding)
            # credacc_cards_status_encoding = predata.credacc_cards_status_52L_encoding
            # data = self.cat_to_int_encode(data, "credacc_cards_status_52L", "credacc_cards_status_encoded", credacc_cards_status_encoding)
            # cacccardblochreas_147M_encoding = predata.cacccardblochreas_147M_encoding
            # data = self.cat_to_int_encode(data, "cacccardblochreas_147M", "cacccardblochreas_encoded", cacccardblochreas_147M_encoding)

            # Adding new columns
            data = data.with_columns(
                # Categorical ordered encoding

                pl.col("conts_type_509L").cast(pl.String).fill_null('None').replace(predata.conts_type_509L_mean_target, default=None).alias("conts_type_mean_target"),
                pl.col("conts_type_509L").cast(pl.String).fill_null('None').replace(predata.conts_type_509L_frequency, default=None).alias("conts_type_frequency"),
                pl.col("conts_type_509L").cast(pl.String).fill_null('None').replace(predata.conts_type_509L_encoding, default=None).alias("conts_type_509L_encoded"),

                pl.col("credacc_cards_status_52L").cast(pl.String).fill_null('None').replace(predata.credacc_cards_status_52L_mean_target, default=None).alias("credacc_cards_status_mean_target"),
                pl.col("credacc_cards_status_52L").cast(pl.String).fill_null('None').replace(predata.credacc_cards_status_52L_frequency, default=None).alias("credacc_cards_status_frequency"),
                pl.col("credacc_cards_status_52L").cast(pl.String).fill_null('None').replace(predata.credacc_cards_status_52L_encoding, default=None).alias("credacc_cards_status_52L_encoded"),

                pl.col("cacccardblochreas_147M").cast(pl.String).fill_null('None').replace(predata.cacccardblochreas_147M_mean_target, default=None).alias("cacccardblochreas_147M_mean_target"),
                pl.col("cacccardblochreas_147M").cast(pl.String).fill_null('None').replace(predata.cacccardblochreas_147M_frequency, default=None).alias("cacccardblochreas_147M_frequency"),
                pl.col("cacccardblochreas_147M").cast(pl.String).fill_null('None').replace(predata.cacccardblochreas_147M_encoding, default=None).alias("cacccardblochreas_147M_encoded"),
            )

            drop_columns = ["conts_type_509L", "credacc_cards_status_52L", "cacccardblochreas_147M"]

        if table_name=='credit_bureau_a_2':

            collater_typofvalofguarant_unique = ['9a0c095e','8fd95e4b','06fb9ba8','3cbe86ba']

            # Adding new columns
            data = data.with_columns(
                pl.col("collater_typofvalofguarant_298M").replace(predata.collater_typofvalofguarant_298M_mean_target, default=None).alias("collater_typofvalofguarant_298M_mean_target"),
                pl.col("collater_typofvalofguarant_298M").replace(predata.collater_typofvalofguarant_298M_frequency, default=None).alias("collater_typofvalofguarant_298M_frequency"),
                pl.col("collater_typofvalofguarant_298M").replace(self.create_ordinal_encoding(predata.collater_typofvalofguarant_298M_mean_target), default=None).alias("collater_typofvalofguarant_298M_encoded"),
                # Add columns as one-hot-encoded values of collater_typofvalofguarant_298M
                #*[pl.col("collater_typofvalofguarant_298M").eq(role).cast(pl.Int16).alias(f"collater_typofvalofguarant_298M_{role}") for role in collater_typofvalofguarant_unique],

                pl.col("collater_typofvalofguarant_407M").replace(predata.collater_typofvalofguarant_407M_mean_target, default=None).alias("collater_typofvalofguarant_407M_mean_target"),
                pl.col("collater_typofvalofguarant_407M").replace(predata.collater_typofvalofguarant_407M_frequency, default=None).alias("collater_typofvalofguarant_407M_frequency"),
                pl.col("collater_typofvalofguarant_407M").replace(self.create_ordinal_encoding(predata.collater_typofvalofguarant_407M_mean_target), default=None).alias("collater_typofvalofguarant_407M_encoded"),
                # Add columns as one-hot-encoded values of collater_typofvalofguarant_407M
                #*[pl.col("collater_typofvalofguarant_407M").eq(role).cast(pl.Int16).alias(f"collater_typofvalofguarant_407M_{role}") for role in collater_typofvalofguarant_unique],

                pl.col("collaterals_typeofguarante_359M").replace(predata.collaterals_typeofguarante_359M_mean_target, default=None).alias("collaterals_typeofguarante_359M_mean_target"),
                pl.col("collaterals_typeofguarante_359M").replace(predata.collaterals_typeofguarante_359M_frequency, default=None).alias("collaterals_typeofguarante_359M_frequency"),
                pl.col("collaterals_typeofguarante_359M").replace(self.create_ordinal_encoding(predata.collaterals_typeofguarante_359M_mean_target), default=None).alias("collaterals_typeofguarante_359M_encoded"),

                pl.col("collaterals_typeofguarante_669M").replace(predata.collaterals_typeofguarante_669M_mean_target, default=None).alias("collaterals_typeofguarante_669M_mean_target"),
                pl.col("collaterals_typeofguarante_669M").replace(predata.collaterals_typeofguarante_669M_frequency, default=None).alias("collaterals_typeofguarante_669M_frequency"),
                pl.col("collaterals_typeofguarante_669M").replace(self.create_ordinal_encoding(predata.collaterals_typeofguarante_669M_mean_target), default=None).alias("collaterals_typeofguarante_669M_encoded"),

                pl.col("subjectroles_name_541M").replace(predata.subjectroles_name_541M_mean_target, default=None).alias("subjectroles_name_541M_mean_target"),
                pl.col("subjectroles_name_541M").replace(predata.subjectroles_name_541M_frequency, default=None).alias("subjectroles_name_541M_frequency"),
                pl.col("subjectroles_name_541M").replace(self.create_ordinal_encoding(predata.subjectroles_name_541M_mean_target), default=None).alias("subjectroles_name_541M_encoded"),

                pl.col("subjectroles_name_838M").replace(predata.subjectroles_name_838M_mean_target, default=None).alias("subjectroles_name_838M_mean_target"),
                pl.col("subjectroles_name_838M").replace(predata.subjectroles_name_838M_frequency, default=None).alias("subjectroles_name_838M_frequency"),
                pl.col("subjectroles_name_838M").replace(self.create_ordinal_encoding(predata.subjectroles_name_838M_mean_target), default=None).alias("subjectroles_name_838M_encoded"),

            )

            drop_columns = ["collater_typofvalofguarant_298M", "collater_typofvalofguarant_407M", "collaterals_typeofguarante_359M", "collaterals_typeofguarante_669M"]

        if table_name=='credit_bureau_b_2':

            data = data.with_columns(
                # Fill null with 0 for pmts_dpdvalue_108P column
                # pl.col("pmts_dpdvalue_108P").fill_null(0).alias("pmts_dpdvalue_108P"),
                # pl.col("pmts_pmtsoverdue_635A").fill_null(0).alias("pmts_pmtsoverdue_635A"),
                pl.col("pmts_dpdvalue_108P").alias("pmts_dpdvalue_108P"),
                pl.col("pmts_pmtsoverdue_635A").alias("pmts_pmtsoverdue_635A"),
            )

        if table_name=='credit_bureau_b_1':
            data = data.with_columns(
                #pl.col("periodicityofpmts_997L").cast(pl.String).fill_null('None').replace(predata.periodicityofpmts_997L_mean_target, default=None).alias("periodicityofpmts_997L_mean_target"),
                pl.col("periodicityofpmts_997L").cast(pl.String).fill_null('None').replace(predata.periodicityofpmts_997L_frequency, default=None).alias("periodicityofpmts_997L_frequency"),
                pl.col("periodicityofpmts_997L").cast(pl.String).fill_null('None').replace(predata.periodicityofpmts_997L_interval, default=None).alias("periodicityofpmts_997L_interval"),
                #pl.col("periodicityofpmts_997L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.periodicityofpmts_997L_mean_target), default=None).alias("periodicityofpmts_997L_encoded"),

                pl.col("classificationofcontr_1114M").replace(predata.classificationofcontr_1114M_mean_target, default=None).alias("classificationofcontr_1114M_mean_target"),
                pl.col("classificationofcontr_1114M").replace(predata.classificationofcontr_1114M_frequency, default=None).alias("classificationofcontr_1114M_frequency"),
                pl.col("classificationofcontr_1114M").replace(self.create_ordinal_encoding(predata.classificationofcontr_1114M_mean_target), default=None).alias("classificationofcontr_1114M_encoded"),

                pl.col("contractst_516M").replace(predata.contractst_516M_mean_target, default=None).alias("contractst_516M_mean_target"),
                pl.col("contractst_516M").replace(predata.contractst_516M_frequency, default=None).alias("contractst_516M_frequency"),
                pl.col("contractst_516M").replace(self.create_ordinal_encoding(predata.contractst_516M_mean_target), default=None).alias("contractst_516M_encoded"),

                pl.col("contracttype_653M").replace(predata.contracttype_653M_mean_target, default=None).alias("contracttype_653M_mean_target"),
                pl.col("contracttype_653M").replace(predata.contracttype_653M_frequency, default=None).alias("contracttype_653M_frequency"),
                pl.col("contracttype_653M").replace(self.create_ordinal_encoding(predata.contracttype_653M_mean_target), default=None).alias("contracttype_653M_encoded"),

                pl.col("credor_3940957M").replace(predata.credor_3940957M_mean_target, default=None).alias("credor_3940957M_mean_target"),
                pl.col("credor_3940957M").replace(predata.credor_3940957M_frequency, default=None).alias("credor_3940957M_frequency"),
                pl.col("credor_3940957M").replace(self.create_ordinal_encoding(predata.credor_3940957M_mean_target), default=None).alias("credor_3940957M_encoded"),

                pl.col("periodicityofpmts_997M").fill_null('None').replace(predata.periodicityofpmts_997M_mean_target, default=None).alias("periodicityofpmts_997M_mean_target"),
                pl.col("periodicityofpmts_997M").fill_null('None').replace(predata.periodicityofpmts_997M_frequency, default=None).alias("periodicityofpmts_997M_frequency"),
                pl.col("periodicityofpmts_997M").fill_null('None').replace(self.create_ordinal_encoding(predata.periodicityofpmts_997M_mean_target), default=None).alias("periodicityofpmts_997M_encoded"),

                
                pl.col("pmtmethod_731M").replace(predata.pmtmethod_731M_mean_target, default=None).alias("pmtmethod_731M_mean_target"),
                pl.col("pmtmethod_731M").replace(predata.pmtmethod_731M_frequency, default=None).alias("pmtmethod_731M_frequency"),
                pl.col("pmtmethod_731M").replace(self.create_ordinal_encoding(predata.pmtmethod_731M_mean_target), default=None).alias("pmtmethod_731M_encoded"),

                pl.col("purposeofcred_722M").replace(predata.purposeofcred_722M_mean_target, default=None).alias("purposeofcred_722M_mean_target"),
                pl.col("purposeofcred_722M").replace(predata.purposeofcred_722M_frequency, default=None).alias("purposeofcred_722M_frequency"),
                pl.col("purposeofcred_722M").replace(self.create_ordinal_encoding(predata.purposeofcred_722M_mean_target), default=None).alias("purposeofcred_722M_encoded"),

                pl.col("subjectrole_326M").replace(predata.subjectrole_326M_mean_target, default=None).alias("subjectrole_326M_mean_target"),
                pl.col("subjectrole_326M").replace(predata.subjectrole_326M_frequency, default=None).alias("subjectrole_326M_frequency"),
                pl.col("subjectrole_326M").replace(self.create_ordinal_encoding(predata.subjectrole_326M_mean_target), default=None).alias("subjectrole_326M_encoded"),

                pl.col("subjectrole_43M").replace(predata.subjectrole_43M_mean_target, default=None).alias("subjectrole_43M_mean_target"),
                pl.col("subjectrole_43M").replace(predata.subjectrole_43M_frequency, default=None).alias("subjectrole_43M_frequency"),
                pl.col("subjectrole_43M").replace(self.create_ordinal_encoding(predata.subjectrole_43M_mean_target), default=None).alias("subjectrole_43M_encoded"),
            )

        if table_name=='credit_bureau_a_1':
            data = data.with_columns(
                pl.col("classificationofcontr_13M").replace(predata.classificationofcontr_13M_mean_target, default=None).alias("classificationofcontr_13M_mean_target"),
                pl.col("classificationofcontr_13M").replace(predata.classificationofcontr_13M_frequency, default=None).alias("classificationofcontr_13M_frequency"),
                pl.col("classificationofcontr_13M").replace(self.create_ordinal_encoding(predata.classificationofcontr_13M_mean_target), default=None).alias("classificationofcontr_13M_encoded"),

                pl.col("classificationofcontr_400M").replace(predata.classificationofcontr_400M_mean_target, default=None).alias("classificationofcontr_400M_mean_target"),
                pl.col("classificationofcontr_400M").replace(predata.classificationofcontr_400M_frequency, default=None).alias("classificationofcontr_400M_frequency"),
                pl.col("classificationofcontr_400M").replace(self.create_ordinal_encoding(predata.classificationofcontr_400M_mean_target), default=None).alias("classificationofcontr_400M_encoded"),

                pl.col("contractst_545M").replace(predata.contractst_545M_mean_target, default=None).alias("contractst_545M_mean_target"),
                pl.col("contractst_545M").replace(predata.contractst_545M_frequency, default=None).alias("contractst_545M_frequency"),
                pl.col("contractst_545M").replace(self.create_ordinal_encoding(predata.contractst_545M_mean_target), default=None).alias("contractst_545M_encoded"),

                pl.col("contractst_964M").replace(predata.contractst_964M_mean_target, default=None).alias("contractst_964M_mean_target"),
                pl.col("contractst_964M").replace(predata.contractst_964M_frequency, default=None).alias("contractst_964M_frequency"),
                 pl.col("contractst_964M").replace(self.create_ordinal_encoding(predata.contractst_964M_mean_target), default=None).alias("contractst_964M_encoded"),

                pl.col("description_351M").replace(predata.description_351M_mean_target, default=None).alias("description_351M_mean_target"),
                pl.col("description_351M").replace(predata.description_351M_frequency, default=None).alias("description_351M_frequency"),
                pl.col("description_351M").replace(self.create_ordinal_encoding(predata.description_351M_mean_target), default=None).alias("description_351M_encoded"),

                pl.col("financialinstitution_382M").replace(predata.financialinstitution_382M_mean_target, default=None).alias("financialinstitution_382M_mean_target"),
                pl.col("financialinstitution_382M").replace(predata.financialinstitution_382M_frequency, default=None).alias("financialinstitution_382M_frequency"),
                pl.col("financialinstitution_382M").replace(self.create_ordinal_encoding(predata.financialinstitution_382M_mean_target), default=None).alias("financialinstitution_382M_encoded"),

                pl.col("financialinstitution_591M").replace(predata.financialinstitution_591M_mean_target, default=None).alias("financialinstitution_591M_mean_target"),
                pl.col("financialinstitution_591M").replace(predata.financialinstitution_591M_frequency, default=None).alias("financialinstitution_591M_frequency"),
                pl.col("financialinstitution_591M").replace(self.create_ordinal_encoding(predata.financialinstitution_591M_mean_target), default=None).alias("financialinstitution_591M_encoded"),

                pl.col("purposeofcred_426M").replace(predata.purposeofcred_426M_mean_target, default=None).alias("purposeofcred_426M_mean_target"),
                pl.col("purposeofcred_426M").replace(predata.purposeofcred_426M_frequency, default=None).alias("purposeofcred_426M_frequency"),
                pl.col("purposeofcred_426M").replace(self.create_ordinal_encoding(predata.purposeofcred_426M_mean_target), default=None).alias("purposeofcred_426M_encoded"),

                pl.col("purposeofcred_874M").replace(predata.purposeofcred_874M_mean_target, default=None).alias("purposeofcred_874M_mean_target"),
                pl.col("purposeofcred_874M").replace(predata.purposeofcred_874M_frequency, default=None).alias("purposeofcred_874M_frequency"),
                pl.col("purposeofcred_874M").replace(self.create_ordinal_encoding(predata.purposeofcred_874M_mean_target), default=None).alias("purposeofcred_874M_encoded"),

                pl.col("subjectrole_182M").replace(predata.subjectrole_182M_mean_target, default=None).alias("subjectrole_182M_mean_target"),
                pl.col("subjectrole_182M").replace(predata.subjectrole_182M_frequency, default=None).alias("subjectrole_182M_frequency"),
                pl.col("subjectrole_182M").replace(self.create_ordinal_encoding(predata.subjectrole_182M_mean_target), default=None).alias("subjectrole_182M_encoded"),

                pl.col("subjectrole_93M").replace(predata.subjectrole_93M_mean_target, default=None).alias("subjectrole_93M_mean_target"),
                pl.col("subjectrole_93M").replace(predata.subjectrole_93M_frequency, default=None).alias("subjectrole_93M_frequency"),
                pl.col("subjectrole_93M").replace(self.create_ordinal_encoding(predata.subjectrole_93M_mean_target), default=None).alias("subjectrole_93M_encoded"),
            )

        if table_name=='person_1':

            persontype_1072L_unique = ['1.0','4.0','5.0']
            education_927M_unique = ['P17_36_170', 'a55475b1', 'P106_81_188', 'P97_36_170', 'P157_18_172', 'P33_146_175']
            empl_employedtotal_800L_unique = ['MORE_FIVE','LESS_ONE','MORE_ONE']
            empl_industry_691L_unique = ['MINING', 'EDUCATION', 'RECRUITMENT', 'TRADE', 'LAWYER', 'ART_MEDIA', 'GOVERNMENT', 'TRANSPORTATION', 'MANUFACTURING', 'MARKETING', 'AGRICULTURE', 'CATERING', 'IT', 'HEALTH', 'WELNESS', 'INSURANCE', 'REAL_ESTATE', 'GAMING', 'ARMY_POLICE', 'POST_TELCO', 'FINANCE', 'OTHER', 'TOURISM', 'CHARITY_RELIGIOUS']
            familystate_447L_unique = ['DIVORCED','WIDOWED','MARRIED','SINGLE', 'LIVING_WITH_PARTNER']
            incometype_1044T_unique = ['SALARIED_GOVT', 'HANDICAPPED_2', 'EMPLOYED', 'PRIVATE_SECTOR_EMPLOYEE', 'SELFEMPLOYED', 'HANDICAPPED', 'RETIRED_PENSIONER', 'HANDICAPPED_3', 'OTHER']
            language1_981M_unique = ['P209_127_106', 'P10_39_147']
            relationshiptoclient_unique = ['SIBLING', 'NEIGHBOR', 'FRIEND', 'OTHER_RELATIVE', 'CHILD', 'OTHER', 'GRAND_PARENT','PARENT', 'SPOUSE', 'COLLEAGUE']
            relationshiptoclient_ordinal_encoding_dict = {'SIBLING': 1, 'NEIGHBOR': 2, 'FRIEND': 3, 'OTHER_RELATIVE': 4, 'CHILD': 5, 'OTHER': 6, 'GRAND_PARENT': 7, 'PARENT': 8, 'SPOUSE': 9, 'COLLEAGUE': 10}

            # Adding new columns
            data = data.with_columns(
                # Each value in the column contaddr_district_15M is a string which has a typical entry as 'P202_53_125'. We transform the data in this column by splitting the string by '_' and keeping the first element
                pl.col("contaddr_district_15M").replace({"a55475b1":None}).str.split_exact('_',n=2).struct.rename_fields(["f1", "f2", "f3"]).alias("fields"),

                pl.col("contaddr_zipcode_807M").replace({"a55475b1":None}).str.split_exact('_',n=2).struct.rename_fields(["z1", "z2", "z3"]).alias("fields_zip"),

                pl.col("empladdr_zipcode_114M").replace({"a55475b1":None}).str.split_exact('_',n=2).struct.rename_fields(["ez1", "ez2", "ez3"]).alias("fields_empzip"),


            ).unnest("fields").unnest("fields_zip").unnest("fields_empzip").drop(
                ["f2", "f3","fields","contaddr_district_15M","z2","z3","fields_zip","contaddr_zipcode_807M","empladdr_zipcode_114M","ez2", "ez3", "fields_empzip"]
            ).rename(
                {
                    "f1": "contaddr_district_15M",
                    "z1": "contaddr_zipcode_807M",
                    "ez1": "empladdr_zipcode_114M",
                }
            ).with_columns(

                # Add columns as one-hot-encoded values of persontype_1072L
                *[pl.col("persontype_1072L").cast(pl.String).eq(perstype).cast(pl.Int16).alias(f"persontype_1072L_{str(perstype)}") for perstype in persontype_1072L_unique],
                pl.col('persontype_1072L').is_null().cast(pl.Int16).alias('persontype_1072L_null'),

                # Add columns as one-hot-encoded values of persontype_792L
                *[pl.col("persontype_792L").cast(pl.String).eq(perstype).cast(pl.Int16).alias(f"persontype_792L_{str(perstype)}") for perstype in persontype_1072L_unique],
                pl.col('persontype_792L').is_null().cast(pl.Int16).alias('persontype_792L_null'),

                # one-hot-encoded values of education_927M
                #*[pl.col("education_927M").eq(edu).cast(pl.Int16).alias(f"education_927M_{edu}") for edu in education_927M_unique],

                # one-hot-encoded for empl_employedtotal_800L
                *[pl.col("empl_employedtotal_800L").cast(pl.String).eq(empl).cast(pl.Int16).alias(f"empl_employedtotal_800L_{empl}") for empl in empl_employedtotal_800L_unique],

                # one-hot-encoded columns for empl_industry_691L
                #*[pl.col("empl_industry_691L").cast(pl.String).eq(empl).cast(pl.Int16).alias(f"empl_industry_691L_{empl}") for empl in empl_industry_691L_unique],
                
                # one-hot-encoded columns for familystate_447L
                #*[pl.col("familystate_447L").cast(pl.String).eq(familystate).cast(pl.Int16).alias(f"familystate_447L_{familystate}") for familystate in familystate_447L_unique],
                #*[pl.col("maritalst_703L").cast(pl.String).eq(familystate).cast(pl.Int16).alias(f"maritalst_703L_{familystate}") for familystate in familystate_447L_unique],

                # TODO: (replaced) one-hot-encoded columns for incometype_1044T
                #*[pl.col("incometype_1044T").cast(pl.String).eq(incometype).cast(pl.Int16).alias(f"incometype_1044T_{incometype}") for incometype in incometype_1044T_unique],

                # one-hot-encoded columns for language1_981M
                *[pl.col("language1_981M").cast(pl.String).eq(language).cast(pl.Int16).alias(f"language1_981M_{language}") for language in language1_981M_unique],

                # TODO: (replaced) one-hot-encoded columns for relationshiptoclient_415T and relationshiptoclient_642T
                #*[pl.col("relationshiptoclient_415T").eq(rel).cast(pl.Int16).alias(f"relationshiptoclient_415T_{rel}") for rel in relationshiptoclient_unique],
                #*[pl.col("relationshiptoclient_642T").eq(rel).cast(pl.Int16).alias(f"relationshiptoclient_642T_{rel}") for rel in relationshiptoclient_unique],
                pl.col("relationshiptoclient_415T").replace(self.create_ordinal_encoding(relationshiptoclient_ordinal_encoding_dict), default=None).alias("relationshiptoclient_415T_encoded"),
                pl.col("relationshiptoclient_642T").replace(self.create_ordinal_encoding(relationshiptoclient_ordinal_encoding_dict), default=None).alias("relationshiptoclient_642T_encoded"),
                


                # one-hot-encoded gender_992L and sex_738L
                #pl.col("gender_992L").eq('M').cast(pl.Int16).alias("gender_992L_M"),
                #pl.col("gender_992L").eq('F').cast(pl.Int16).alias("gender_992L_F"),
                #pl.col("gender_992L").is_null().cast(pl.Int16).alias("gender_992L_null"),
                #pl.col("sex_738L").eq('M').cast(pl.Int16).alias("sex_738L_M"),
                #pl.col("sex_738L").eq('F').cast(pl.Int16).alias("sex_738L_F"),


                # Mean target and frequency (TODO: error on these because of replace)
                #pl.col("persontype_1072L").fill_null('None').replace(predata.persontype_1072L_mean_target, default=None).alias("persontype_1072L_mean_target"),
                #pl.col("persontype_1072L").fill_null('None').replace(predata.persontype_1072L_frequency, default=None).alias("persontype_1072L_frequency"),
                #pl.col("persontype_792L").fill_null('None').replace(predata.persontype_792L_mean_target, default=None).alias("persontype_792L_mean_target"),
                #pl.col("persontype_792L").fill_null('None').replace(predata.persontype_792L_frequency, default=None).alias("persontype_792L_frequency"),

                # Mean target and frequency for truncated columns
                pl.col("contaddr_district_15M").replace(predata.contaddr_district_15M_mean_target, default=None).cast(pl.Float64).alias("contaddr_district_15M_mean_target"),
                pl.col("contaddr_district_15M").replace(predata.contaddr_district_15M_frequency, default=None).cast(pl.Int64).cast(pl.Int64).alias("contaddr_district_15M_frequency"),
                pl.col("contaddr_district_15M").replace(self.create_ordinal_encoding(predata.contaddr_district_15M_mean_target), default=None).alias("contaddr_district_15M_encoded"),

                pl.col("contaddr_zipcode_807M").replace(predata.contaddr_zipcode_807M_mean_target, default=None).cast(pl.Float64).alias("contaddr_zipcode_807M_mean_target"),
                pl.col("contaddr_zipcode_807M").replace(predata.contaddr_zipcode_807M_frequency, default=None).cast(pl.Int64).alias("contaddr_zipcode_807M_frequency"),
                pl.col("contaddr_zipcode_807M").replace(self.create_ordinal_encoding(predata.contaddr_zipcode_807M_mean_target), default=None).alias("contaddr_zipcode_807M_encoded"),

                pl.col("empladdr_zipcode_114M").replace(predata.empladdr_zipcode_114M_mean_target, default=None).cast(pl.Float64).alias("empladdr_zipcode_114M_mean_target"),
                pl.col("empladdr_zipcode_114M").replace(predata.empladdr_zipcode_114M_frequency, default=None).cast(pl.Int64).alias("empladdr_zipcode_114M_frequency"),
                pl.col("empladdr_zipcode_114M").replace(self.create_ordinal_encoding(predata.empladdr_zipcode_114M_mean_target), default=None).alias("empladdr_zipcode_114M_encoded"),

                # Mean target and frequency
                pl.col("education_927M").replace(predata.education_927M_mean_target, default=None).cast(pl.Float64).alias("education_927M_mean_target"),
                pl.col("education_927M").replace(predata.education_927M_frequency, default=None).cast(pl.Int64).alias("education_927M_frequency"),
                pl.col("education_927M").replace(self.create_ordinal_encoding(predata.education_927M_mean_target), default=None).cast(pl.Int32).alias("education_927M_encoded"),

                pl.col("empl_employedtotal_800L").cast(pl.String).fill_null('None').replace(predata.empl_employedtotal_800L_mean_target, default=None).cast(pl.Float64).alias("empl_employedtotal_800L_mean_target"),
                pl.col("empl_employedtotal_800L").cast(pl.String).fill_null('None').replace(predata.empl_employedtotal_800L_frequency, default=None).cast(pl.Int64).alias("empl_employedtotal_800L_frequency"),
                pl.col("empl_employedtotal_800L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.empl_employedtotal_800L_mean_target), default=None).cast(pl.Int32).alias("empl_employedtotal_800L_encoded"),

                pl.col("empl_industry_691L").cast(pl.String).fill_null('None').replace(predata.empl_industry_691L_mean_target, default=None).cast(pl.Float64).alias("empl_industry_691L_mean_target"),
                pl.col("empl_industry_691L").cast(pl.String).fill_null('None').replace(predata.empl_industry_691L_frequency, default=None).cast(pl.Int64).alias("empl_industry_691L_frequency"),
                pl.col("empl_industry_691L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.empl_industry_691L_mean_target), default=None).cast(pl.Int32).alias("empl_industry_691L_encoded"),

                pl.col("empladdr_district_926M").replace(predata.empladdr_district_926M_mean_target, default=None).cast(pl.Float64).alias("empladdr_district_926M_mean_target"),
                pl.col("empladdr_district_926M").replace(predata.empladdr_district_926M_frequency, default=None).cast(pl.Int64).alias("empladdr_district_926M_frequency"),
                pl.col("empladdr_district_926M").replace(self.create_ordinal_encoding(predata.empladdr_district_926M_mean_target), default=None).cast(pl.Int32).alias("empladdr_district_926M_encoded"),
                
                pl.col("familystate_447L").cast(pl.String).fill_null('None').replace(predata.familystate_447L_mean_target, default=None).cast(pl.Float64).alias("familystate_447L_mean_target"),
                pl.col("familystate_447L").cast(pl.String).fill_null('None').replace(predata.familystate_447L_frequency, default=None).cast(pl.Int64).alias("familystate_447L_frequency"),
                pl.col("familystate_447L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.familystate_447L_mean_target), default=None).cast(pl.Int32).alias("familystate_447L_encoded"),
                pl.col("maritalst_703L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.familystate_447L_mean_target), default=None).cast(pl.Int32).alias("maritalst_703L_encoded"),
                
                pl.col("housetype_905L").cast(pl.String).fill_null('None').replace(predata.housetype_905L_mean_target, default=None).cast(pl.Float64).alias("housetype_905L_mean_target"),
                #pl.col("housetype_905L").cast(pl.String).fill_null('None').replace(predata.housetype_905L_frequency, default=None).cast(pl.Int64).alias("housetype_905L_frequency"),
                pl.col("housetype_905L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.housetype_905L_mean_target), default=None).cast(pl.Int32).alias("housetype_905L_encoded"),

                pl.col("housingtype_772L").cast(pl.String).fill_null('None').replace(predata.housingtype_772L_mean_target, default=None).cast(pl.Float64).alias("housingtype_772L_mean_target"),
                #pl.col("housingtype_772L").cast(pl.String).fill_null('None').replace(predata.housingtype_772L_frequency, default=None).cast(pl.Int64).alias("housingtype_772L_frequency"),
                pl.col("housingtype_772L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.housingtype_772L_mean_target), default=None).cast(pl.Int32).alias("housingtype_772L_encoded"),

                pl.col("incometype_1044T").fill_null('None').replace(predata.incometype_1044T_mean_target, default=None).cast(pl.Float64).alias("incometype_1044T_mean_target"),
                pl.col("incometype_1044T").fill_null('None').replace(predata.incometype_1044T_frequency, default=None).cast(pl.Int64).alias("incometype_1044T_frequency"),
                pl.col("incometype_1044T").fill_null('None').replace(self.create_ordinal_encoding(predata.incometype_1044T_mean_target), default=None).cast(pl.Int32).alias("incometype_1044T_encoded"),

                pl.col("language1_981M").fill_null('None').replace(predata.language1_981M_mean_target, default=None).cast(pl.Float64).alias("language1_981M_mean_target"),
                pl.col("language1_981M").fill_null('None').replace(predata.language1_981M_frequency, default=None).cast(pl.Int64).alias("language1_981M_frequency"),
                pl.col("language1_981M").fill_null('None').replace(self.create_ordinal_encoding(predata.language1_981M_mean_target), default=None).cast(pl.Int32).alias("language1_981M_encoded"),


                pl.col("role_1084L").cast(pl.String).fill_null('None').replace(predata.role_1084L_mean_target, default=None).cast(pl.Float64).alias("role_1084L_mean_target"),
                #pl.col("role_1084L").cast(pl.String).fill_null('None').replace(predata.role_1084L_frequency, default=None).cast(pl.Int64).alias("role_1084L_frequency"),
                pl.col("role_1084L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.role_1084L_mean_target), default=None).cast(pl.Int32).alias("role_1084L_encoded"),


                pl.col("type_25L").cast(pl.String).fill_null('None').replace(predata.type_25L_mean_target, default=None).cast(pl.Float64).alias("type_25L_mean_target"),
                pl.col("type_25L").cast(pl.String).fill_null('None').replace(predata.type_25L_frequency, default=None).cast(pl.Int64).alias("type_25L_frequency"),
                pl.col("type_25L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.type_25L_mean_target), default=None).cast(pl.Int32).alias("type_25L_encoded"),

                # Categorical (many categories to int)
                pl.col("registaddr_district_1083M").replace(predata.registaddr_district_1083M_idx, default=0).cast(pl.Int32).alias("registaddr_district_1083M"),

            ).with_columns(
                # average between housetype_905L_mean_target and housetype_905L_mean_target
                (pl.col("housetype_905L_mean_target").fill_null(0.0) + pl.col("housingtype_772L_mean_target").fill_null(0.0)).mul(0.5).cast(pl.Float64).alias("housetype_905L_772L_mean_target"),
                
            ).drop(["housetype_905L_mean_target","housetype_905L_mean_target"])

        if table_name=='applprev_1':

            credtype_587L_unique = ['REL','CAL','COL']

            # Adding new columns
            data = data.with_columns(
                pl.col("cancelreason_3545846M").cast(pl.String).replace(predata.cancelreason_3545846M_mean_target, default=None).cast(pl.Float64).alias("cancelreason_3545846M_mean_target"),
                pl.col("cancelreason_3545846M").cast(pl.String).replace(predata.cancelreason_3545846M_frequency, default=None).cast(pl.Int64).alias("cancelreason_3545846M_frequency"),
                pl.col("cancelreason_3545846M").cast(pl.String).replace(self.create_ordinal_encoding(predata.cancelreason_3545846M_mean_target), default=None).cast(pl.Int32).alias("cancelreason_3545846M_encoded"),

                pl.col("credacc_status_367L").cast(pl.String).fill_null('None').replace(predata.credacc_status_367L_mean_target, default=None).cast(pl.Float64).alias("credacc_status_367L_mean_target"),
                pl.col("credacc_status_367L").cast(pl.String).fill_null('None').replace(predata.credacc_status_367L_frequency, default=None).cast(pl.Int64).alias("credacc_status_367L_frequency"),
                pl.col("credacc_status_367L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.credacc_status_367L_mean_target), default=None).cast(pl.Int32).alias("credacc_status_367L_encoded"),

                pl.col("credtype_587L").cast(pl.String).fill_null('None').replace(predata.credtype_587L_mean_target, default=None).cast(pl.Float64).alias("credtype_587L_mean_target"),
                pl.col("credtype_587L").cast(pl.String).fill_null('None').replace(predata.credtype_587L_frequency, default=None).cast(pl.Int64).alias("credtype_587L_frequency"),
                #pl.col("credtype_587L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.credtype_587L_mean_target), default=None).cast(pl.Int32).alias("credtype_587L_encoded"),


                # TODO: define one-hot-encoded
                pl.col("education_1138M").cast(pl.String).replace(predata.education_1138M_mean_target, default=None).cast(pl.Float64).alias("education_1138M_mean_target"),
                pl.col("education_1138M").cast(pl.String).replace(predata.education_1138M_frequency, default=None).cast(pl.Int64).alias("education_1138M_frequency"),
                pl.col("education_1138M").cast(pl.String).replace(self.create_ordinal_encoding(predata.education_1138M_mean_target), default=None).cast(pl.Int32).alias("education_1138M_encoded"),

                pl.col("familystate_726L").cast(pl.String).fill_null('None').replace(predata.familystate_726L_mean_target, default=None).cast(pl.Float64).alias("familystate_726L_mean_target"),
                pl.col("familystate_726L").cast(pl.String).fill_null('None').replace(predata.familystate_726L_frequency, default=None).cast(pl.Int64).alias("familystate_726L_frequency"),
                pl.col("familystate_726L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.familystate_726L_mean_target), default=None).cast(pl.Int32).alias("familystate_726L_encoded"),

                pl.col("inittransactioncode_279L").cast(pl.String).fill_null('None').replace(predata.inittransactioncode_279L_mean_target, default=None).cast(pl.Float64).alias("inittransactioncode_279L_mean_target"),
                pl.col("inittransactioncode_279L").cast(pl.String).fill_null('None').replace(predata.inittransactioncode_279L_frequency, default=None).cast(pl.Int64).alias("inittransactioncode_279L_frequency"),
                pl.col("inittransactioncode_279L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.inittransactioncode_279L_mean_target), default=None).cast(pl.Int32).alias("inittransactioncode_279L_encoded"),

                pl.col("postype_4733339M").replace(predata.postype_4733339M_mean_target, default=None).cast(pl.Float64).alias("postype_4733339M_mean_target"),
                pl.col("postype_4733339M").replace(predata.postype_4733339M_frequency, default=None).cast(pl.Int64).alias("postype_4733339M_frequency"),
                pl.col("postype_4733339M").replace(self.create_ordinal_encoding(predata.postype_4733339M_mean_target), default=None).cast(pl.Int32).alias("postype_4733339M_encoded"),

                pl.col("rejectreason_755M").replace(predata.rejectreason_755M_mean_target, default=None).cast(pl.Float64).alias("rejectreason_755M_mean_target"),
                pl.col("rejectreason_755M").replace(predata.rejectreason_755M_frequency, default=None).cast(pl.Int64).alias("rejectreason_755M_frequency"),
                pl.col("rejectreason_755M").replace(self.create_ordinal_encoding(predata.rejectreason_755M_mean_target), default=None).cast(pl.Int32).alias("rejectreason_755M_encoded"),

                pl.col("rejectreasonclient_4145042M").replace(predata.rejectreasonclient_4145042M_mean_target, default=None).cast(pl.Float64).alias("rejectreasonclient_4145042M_mean_target"),
                pl.col("rejectreasonclient_4145042M").replace(predata.rejectreasonclient_4145042M_frequency, default=None).cast(pl.Int64).alias("rejectreasonclient_4145042M_frequency"),
                pl.col("rejectreasonclient_4145042M").replace(self.create_ordinal_encoding(predata.rejectreasonclient_4145042M_mean_target), default=None).cast(pl.Int32).alias("rejectreasonclient_4145042M_encoded"),

                pl.col("status_219L").cast(pl.String).fill_null('None').replace(predata.status_219L_mean_target, default=None).cast(pl.Float64).alias("status_219L_mean_target"),
                pl.col("status_219L").cast(pl.String).fill_null('None').replace(predata.status_219L_frequency, default=None).cast(pl.Int64).alias("status_219L_frequency"),
                pl.col("status_219L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.status_219L_mean_target), default=None).cast(pl.Int32).alias("status_219L_encoded"),


                # one-hot-encoded for credtype_587L
                *[pl.col("credtype_587L").cast(pl.String).eq(credtype).cast(pl.Int16).alias(f"credtype_587L_{credtype}") for credtype in credtype_587L_unique],

            )

        if table_name=='static_0':

            cardtype_51L_unique = ['PERSONALIZED', 'INSTANT']
            credtype_322L_unique = ['REL', 'CAL', 'COL']

            droplist = ['twobodfilling_608L','cardtype_51L','disbursementtype_67L','inittransactioncode_186L','lastapprcommoditycat_1041M','lastapprcommoditytypec_5251766M',
                    'lastcancelreason_561M','lastrejectcommoditycat_161M','lastrejectcommodtypec_5251769M','lastrejectreason_759M','lastrejectreason_759M',
                    'lastrejectreasonclient_4145040M','lastrejectreasonclient_4145040M','lastst_736L','previouscontdistrict_112M', 'credtype_322L']
            
            
            bool_columns = ['equalitydataagreement_891L','equalityempfrom_62L','isbidproduct_1095L','isbidproductrequest_292L','isdebitcard_729L','opencred_647L']

            # Adding new columns
            data = data.with_columns(

                ###### pl.String (Categorical)

                pl.col("bankacctype_710L").cast(pl.String).eq('CA').cast(pl.Int16).alias("bankacctype_710L"),
                pl.col("paytype1st_925L").cast(pl.String).eq('OTHER').cast(pl.Int16).alias("paytype1st_925L"),
                pl.col("paytype_783L").cast(pl.String).eq('OTHER').cast(pl.Int16).alias("paytype_783L"),

                pl.col("twobodfilling_608L").cast(pl.String).eq('BO').cast(pl.Int16).alias("twobodfilling_608L_BO"),
                pl.col("twobodfilling_608L").cast(pl.String).eq('FO').cast(pl.Int16).alias("twobodfilling_608L_FO"),

                pl.col("typesuite_864L").cast(pl.String).eq('AL').cast(pl.Int16).alias("typesuite_864L"),
                
                pl.col("credtype_322L").cast(pl.String).fill_null('None').replace(predata.credtype_322L_mean_target, default=None).cast(pl.Float64).alias("credtype_322L_mean_target"),
                #pl.col("credtype_322L").cast(pl.String).fill_null('None').replace(predata.credtype_322L_frequency, default=None).cast(pl.Int64).alias("credtype_322L_frequency"),
                #pl.col("credtype_322L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.credtype_322L_mean_target), default=None).cast(pl.Int32).alias("credtype_322L_encoded"),

                pl.col("cardtype_51L").cast(pl.String).fill_null('None').replace(predata.cardtype_51L_mean_target, default=None).cast(pl.Float64).alias("cardtype_51L_mean_target"),
                #pl.col("cardtype_51L").cast(pl.String).fill_null('None').replace(predata.cardtype_51L_frequency, default=None).cast(pl.Int64).alias("cardtype_51L_frequency"),
                #pl.col("cardtype_51L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.cardtype_51L_mean_target), default=None).cast(pl.Int32).alias("cardtype_51L_encoded"),

                pl.col("disbursementtype_67L").cast(pl.String).fill_null('None').replace(predata.disbursementtype_67L_mean_target, default=None).cast(pl.Float64).alias("disbursementtype_67L_mean_target"),
                #pl.col("disbursementtype_67L").cast(pl.String).fill_null('None').replace(predata.disbursementtype_67L_frequency, default=None).cast(pl.Int64).alias("disbursementtype_67L_frequency"),
                pl.col("disbursementtype_67L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.disbursementtype_67L_mean_target), default=None).cast(pl.Int32).alias("disbursementtype_67L_encoded"),

                pl.col("inittransactioncode_186L").cast(pl.String).fill_null('None').replace(predata.inittransactioncode_186L_mean_target, default=None).cast(pl.Float64).alias("inittransactioncode_186L_mean_target"),
                #pl.col("inittransactioncode_186L").cast(pl.String).fill_null('None').replace(predata.inittransactioncode_186L_frequency, default=None).cast(pl.Int64).alias("inittransactioncode_186L_frequency"),
                pl.col("inittransactioncode_186L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.inittransactioncode_186L_mean_target), default=None).cast(pl.Int32).alias("inittransactioncode_186L_encoded"),

                pl.col("lastapprcommoditycat_1041M").replace(predata.lastapprcommoditycat_1041M_mean_target, default=None).cast(pl.Float64).alias("lastapprcommoditycat_1041M_mean_target"),
                #pl.col("lastapprcommoditycat_1041M").replace(predata.lastapprcommoditycat_1041M_frequency, default=None).cast(pl.Int64).alias("lastapprcommoditycat_1041M_frequency"),
                pl.col("lastapprcommoditycat_1041M").replace(self.create_ordinal_encoding(predata.lastapprcommoditycat_1041M_mean_target), default=None).cast(pl.Int32).alias("lastapprcommoditycat_1041M_encoded"),

                pl.col("lastapprcommoditytypec_5251766M").replace(predata.lastapprcommoditytypec_5251766M_mean_target, default=None).cast(pl.Float64).alias("lastapprcommoditytypec_5251766M_mean_target"),
                #pl.col("lastapprcommoditytypec_5251766M").replace(predata.lastapprcommoditytypec_5251766M_frequency, default=None).cast(pl.Int64).alias("lastapprcommoditytypec_5251766M_frequency"),
                pl.col("lastapprcommoditytypec_5251766M").replace(self.create_ordinal_encoding(predata.lastapprcommoditytypec_5251766M_mean_target), default=None).cast(pl.Int32).alias("lastapprcommoditytypec_5251766M_encoded"),

                pl.col("lastcancelreason_561M").replace(predata.lastcancelreason_561M_mean_target, default=None).cast(pl.Float64).alias("lastcancelreason_561M_mean_target"),
                #pl.col("lastcancelreason_561M").replace(predata.lastcancelreason_561M_frequency, default=None).cast(pl.Int64).alias("lastcancelreason_561M_frequency"),
                pl.col("lastcancelreason_561M").replace(self.create_ordinal_encoding(predata.lastcancelreason_561M_mean_target), default=None).cast(pl.Int32).alias("lastcancelreason_561M_encoded"),

                pl.col("lastrejectcommoditycat_161M").replace(predata.lastrejectcommoditycat_161M_mean_target, default=None).cast(pl.Float64).alias("lastrejectcommoditycat_161M_mean_target"),
                #pl.col("lastrejectcommoditycat_161M").replace(predata.lastrejectcommoditycat_161M_frequency, default=None).cast(pl.Int64).alias("lastrejectcommoditycat_161M_frequency"),
                pl.col("lastrejectcommoditycat_161M").replace(self.create_ordinal_encoding(predata.lastrejectcommoditycat_161M_mean_target), default=None).cast(pl.Int32).alias("lastrejectcommoditycat_161M_encoded"),

                pl.col("lastrejectcommodtypec_5251769M").replace(predata.lastrejectcommodtypec_5251769M_mean_target, default=None).cast(pl.Float64).alias("lastrejectcommodtypec_5251769M_mean_target"),
                #pl.col("lastrejectcommodtypec_5251769M").replace(predata.lastrejectcommodtypec_5251769M_frequency, default=None).cast(pl.Int64).alias("lastrejectcommodtypec_5251769M_frequency"),
                pl.col("lastrejectcommodtypec_5251769M").replace(self.create_ordinal_encoding(predata.lastrejectcommodtypec_5251769M_mean_target), default=None).cast(pl.Int32).alias("lastrejectcommodtypec_5251769M_encoded"),

                pl.col("lastrejectreason_759M").replace(predata.lastrejectreason_759M_mean_target, default=None).cast(pl.Float64).alias("lastrejectreason_759M_mean_target"),
                #pl.col("lastrejectreason_759M").replace(predata.lastrejectreason_759M_frequency, default=None).cast(pl.Int64).alias("lastrejectreason_759M_frequency"),
                pl.col("lastrejectreason_759M").replace(self.create_ordinal_encoding(predata.lastrejectreason_759M_mean_target), default=None).cast(pl.Int32).alias("lastrejectreason_759M_encoded"),

                pl.col("lastrejectreasonclient_4145040M").replace(predata.lastrejectreasonclient_4145040M_mean_target, default=None).cast(pl.Float64).alias("lastrejectreasonclient_4145040M_mean_target"),
                #pl.col("lastrejectreasonclient_4145040M").replace(predata.lastrejectreasonclient_4145040M_frequency, default=None).cast(pl.Int64).alias("lastrejectreasonclient_4145040M_frequency"),
                pl.col("lastrejectreasonclient_4145040M").replace(self.create_ordinal_encoding(predata.lastrejectreasonclient_4145040M_mean_target), default=None).cast(pl.Int32).alias("lastrejectreasonclient_4145040M_encoded"),

                
                pl.col("lastst_736L").cast(pl.String).fill_null('None').replace(predata.lastst_736L_mean_target, default=None).cast(pl.Float64).alias("lastst_736L_mean_target"),
                #pl.col("lastst_736L").cast(pl.String).fill_null('None').replace(predata.lastst_736L_frequency, default=None).cast(pl.Int64).alias("lastst_736L_frequency"),
                pl.col("lastst_736L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.lastst_736L_mean_target), default=None).cast(pl.Int32).alias("lastst_736L_encoded"),


                pl.col("previouscontdistrict_112M").replace(predata.previouscontdistrict_112M_mean_target, default=None).cast(pl.Float64).alias("previouscontdistrict_112M_mean_target"),
                #pl.col("previouscontdistrict_112M").replace(predata.previouscontdistrict_112M_frequency, default=None).cast(pl.Int64).alias("previouscontdistrict_112M_frequency"),
                pl.col("previouscontdistrict_112M").replace(self.create_ordinal_encoding(predata.previouscontdistrict_112M_mean_target), default=None).cast(pl.Int32).alias("previouscontdistrict_112M_encoded"),

                # one-hot-encoded columns for cardtype_51L
                *[pl.col("cardtype_51L").cast(pl.String).eq(cardtype).cast(pl.Int16).alias(f"cardtype_51L_{cardtype}") for cardtype in cardtype_51L_unique],

                # one-hot-encoded columns for credtype_322L
                *[pl.col("credtype_322L").cast(pl.String).eq(credtype).cast(pl.Int16).alias(f"credtype_322L_{credtype}") for credtype in credtype_322L_unique],

                ###### pl.Boolean
                #*[pl.col(col).cast(pl.Int8, strict=False).fill_null(-1).alias(col) for col in bool_columns],
                *[pl.col(col).cast(pl.Int8, strict=False).alias(col) for col in bool_columns],

                ###### pl.Date
                *[pl.col(col).alias(col) for col in predata.date_static_0_columns],

                ###### Numeric (TODO: split into Float and int?)
                #*[pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col) for col in predata.numeric_static_0],
                *[pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in predata.numeric_static_0],

                # case_id
                pl.col('case_id').cast(pl.Int64).alias('case_id'),

            ).drop(droplist)

        if table_name=='static_cb_0':
            # Adding new columns
            data = data.with_columns(
                # Date of birth: combined birthdate_574D, dateofbirth_337D and dateofbirth_342D
                # TODO: combine with birth_259D_87D from other table
                pl.max_horizontal(pl.col('birthdate_574D'), pl.col('dateofbirth_337D'), pl.col('dateofbirth_342D')).alias('dateofbirth_574D_337D_342D'),

                # Combine assignment date: assignmentdate_238D and assignmentdate_4527235D
                pl.max_horizontal(pl.col('assignmentdate_238D'), pl.col('assignmentdate_4527235D'), pl.col('assignmentdate_4955616D')).alias('assignmentdate_238D_4527235D_4955616D'),

                # Combine response date: responsedate_1012D, responsedate_4527233D, responsedate_4917613D
                pl.max_horizontal(pl.col('responsedate_1012D'), pl.col('responsedate_4527233D'), pl.col('responsedate_4917613D')).alias('responsedate_1012D_4527233D_4917613D'),

                #### pl.String columns

                # Create two binary variables whether the description_5085714M is equal to '2fc785b2' or 'a55475b1'
                pl.col("description_5085714M").eq('2fc785b2').cast(pl.Int16).alias("description_5085714M_2fc785b2"),
                pl.col("description_5085714M").eq('a55475b1').cast(pl.Int16).alias("description_5085714M_a55475b1"),
 
                
                # targets
                pl.col("requesttype_4525192L").cast(pl.String).fill_null('None').replace(predata.requesttype_4525192L_mean_target, default=None).cast(pl.Float64).alias("requesttype_4525192L_mean_target"),
                pl.col("requesttype_4525192L").cast(pl.String).fill_null('None').replace(predata.requesttype_4525192L_frequency, default=None).cast(pl.Int64).alias("requesttype_4525192L_frequency"),
                pl.col("requesttype_4525192L").cast(pl.String).fill_null('None').replace(self.create_ordinal_encoding(predata.requesttype_4525192L_mean_target), default=None).cast(pl.Int32).alias("requesttype_4525192L_encoded"),

                pl.col("education_1103M").replace(predata.education_1103M_mean_target, default=None).cast(pl.Float64).alias("education_1103M_mean_target"),
                #pl.col("education_1103M").replace(predata.education_1103M_frequency, default=None).cast(pl.Int64).alias("education_1103M_frequency"),
                pl.col("education_1103M").replace(self.create_ordinal_encoding(predata.education_1103M_mean_target), default=None).cast(pl.Int32).alias("education_1103M_encoded"),

                pl.col("education_88M").replace(predata.education_88M_mean_target, default=None).cast(pl.Float64).alias("education_88M_mean_target"),
                #pl.col("education_88M").replace(predata.education_88M_frequency, default=None).cast(pl.Int64).alias("education_88M_frequency"),
                pl.col("education_88M").replace(self.create_ordinal_encoding(predata.education_88M_mean_target), default=None).cast(pl.Int32).alias("education_88M_encoded"),

                pl.col("maritalst_385M").replace(predata.maritalst_385M_mean_target, default=None).cast(pl.Float64).alias("maritalst_385M_mean_target"),
                #pl.col("maritalst_385M").replace(predata.maritalst_385M_frequency, default=None).cast(pl.Int64).alias("maritalst_385M_frequency"),
                pl.col("maritalst_385M").replace(self.create_ordinal_encoding(predata.maritalst_385M_mean_target), default=None).cast(pl.Int32).alias("maritalst_385M_encoded"),

                pl.col("maritalst_893M").replace(predata.maritalst_893M_mean_target, default=None).cast(pl.Float64).alias("maritalst_893M_mean_target"),
                #pl.col("maritalst_893M").replace(predata.maritalst_893M_frequency, default=None).cast(pl.Int64).alias("maritalst_893M_frequency"),
                pl.col("maritalst_893M").replace(self.create_ordinal_encoding(predata.maritalst_893M_mean_target), default=None).cast(pl.Int32).alias("maritalst_893M_encoded"),

                # TODO: try one-hot encoding! (done)
                pl.col('riskassesment_302T').cast(pl.String).fill_null('None').replace(predata.riskassesment_302T_mean_target, default=None).cast(pl.Float64).alias("riskassesment_302T_mean_target"),
                #pl.col("riskassesment_302T").cast(pl.String).fill_null('None').replace(predata.riskassesment_302T_frequency, default=None).cast(pl.Int64).alias("riskassesment_302T_frequency"),
                pl.col("riskassesment_302T").cast(pl.String).fill_null('None').replace(predata.riskassesment_302T_probability, default=None).cast(pl.Float64).alias("riskassesment_302T_probability"),
                # One-hot-encoding for riskassesment_302T
                *[pl.col("riskassesment_302T").cast(pl.String).fill_null('None').eq(riskassesment).cast(pl.Int16).alias(f"riskassesment_302T_{predata.riskassesment_302T_categorical_encoding[riskassesment]}") for riskassesment in predata.riskassesment_302T_unique],


                # riskassesment_940T: String to Float
                pl.col('riskassesment_940T').cast(pl.Float64, strict=False).alias('riskassesment_940T'),

                ###### Numeric (TODO: split into Float and int?)
                #*[pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col) for col in predata.numeric_static_cb_0],
                *[pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in predata.numeric_static_cb_0],


            ).with_columns(
                # Difference between assignment date and response date (in days)
                (pl.col("assignmentdate_238D_4527235D_4955616D") - pl.col("responsedate_1012D_4527233D_4917613D")).dt.total_days().alias("assignmentdate_238D_responsedate_1012D_duration"),

                # Age of a person: <25, 25-35, 35-45, 45-60, >60
                pl.when(
                    (pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).le(25.0)
                ).then(1).otherwise(0).cast(pl.Int16).alias("age_25"),
                pl.when(
                   ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).gt(25.0)) & ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).le(35.0))
                ).then(1).otherwise(0).cast(pl.Int16).alias("age_25_35"),
                pl.when(
                   ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).gt(35.0)) & ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).le(45.0))
                ).then(1).otherwise(0).cast(pl.Int16).alias("age_35_45"),
                pl.when(
                   ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).gt(45.0)) & ((pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).le(60.0))
                ).then(1).otherwise(0).cast(pl.Int16).alias("age_45_60"),
                pl.when(
                   (pl.col('assignmentdate_238D_4527235D_4955616D') - pl.col('dateofbirth_574D_337D_342D')).dt.total_days().mul(1.0/365).gt(60.0)
                ).then(1).otherwise(0).cast(pl.Int16).alias("age_60"),

            ).drop(predata.static_cb_0_dropcols)


        return data
    
    
    def aggregate_depth_2(self, data: pl.DataFrame, table_name: str) -> pl.DataFrame:
        """
        Aggregate the data by depth 2:
        - data: DataFrame to aggregate
        - table_name: name of the table to aggregate
        - return: aggregated DataFrame
        """

        # Create a new DataFrame to store the aggregated results
        #result = pl.DataFrame()

        if table_name=='person_2':
            # Encode categorical columns
            #data = self.encode_categorical_columns(data, table_name)
            
            data = data.group_by(['case_id','num_group1']).agg(
                    # Number of non-null related person roles indicated
                    pl.when(pl.col("relatedpersons_role_762T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_relatedpersons_role_762T"),
                    # The most influential role (cat)
                    pl.col("relatedpersons_role_762T_encoded").max().alias("relatedpersons_role_762T_encoded_max"),
                    # Start date of employment
                    pl.col("empls_employedfrom_796D").drop_nulls().first().alias('empls_employedfrom_796D'),
                    # Various mean_target columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                    # Various frequency columns
                    *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                    # Various ordinal encoded columns
                    *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                    *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],


                )
            
        elif table_name=='applprev_2':
            # Encode categorical columns
            #data = self.encode_categorical_columns(data, table_name)

            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_group2', '_id']

            data = data.group_by(['case_id','num_group1']).agg(
                # Number of non-null contact roles indicated
                pl.when(pl.col("conts_type_509L_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_conts_type_509L_encoded"),
                # The most influential contact
                pl.col("conts_type_509L_encoded").drop_nulls().max().alias("conts_type_509L_encoded_max"),

                # Number of non-null credacc_cards_status
                # TODO: check .replace("a55475b1", None) addition
                pl.when(pl.col("credacc_cards_status_52L_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_credacc_cards_status_52L_encoded"),
                # The most influential credacc_cards_status
                pl.col("credacc_cards_status_52L_encoded").max().alias("credacc_cards_status_52L_encoded_max"),

                # Number of credit card blocks
                # TODO: check .replace("a55475b1", None) addition
                pl.when(pl.col("cacccardblochreas_147M_encoded").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_cacccardblochreas_147M_encoded"),
                # The most influential credit card block
                pl.col("cacccardblochreas_147M_encoded").max().alias("cacccardblochreas_147M_encoded_max"),

                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],

                # Pick last three values of groupby and compute mean for all numerical columns
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                
            )
            
        elif table_name=='credit_bureau_a_2':
            # Encode categorical columns
            #data = self.encode_categorical_columns(data, table_name)

            collater_typofvalofguarant_unique = ['9a0c095e','8fd95e4b','06fb9ba8','3cbe86ba']
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_group2', '_id']

            data = data.group_by(['case_id', 'num_group1']).agg(
                # Number of non-null collater_typofvalofguarant_298M
                pl.when(pl.col("collater_typofvalofguarant_298M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collater_typofvalofguarant_298M"),
                # Sum of one-hot-encoded columns
                #*[pl.col(f"collater_typofvalofguarant_298M_{role}").sum().cast(pl.Int16).alias(f"collater_typofvalofguarant_{role}_298M") for role in collater_typofvalofguarant_unique],

                # Number of non-null collater_typofvalofguarant_407M
                pl.when(pl.col("collater_typofvalofguarant_407M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collater_typofvalofguarant_407M"),
                # Sum of one-hot-encoded columns
                #*[pl.col(f"collater_typofvalofguarant_407M_{role}").sum().cast(pl.Int16).alias(f"collater_typofvalofguarant_{role}_407M") for role in collater_typofvalofguarant_unique],

                # Number of non-null collater_valueofguarantee_1124L
                pl.when(pl.col("collater_valueofguarantee_1124L").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collater_valueofguarantee_1124L"),
                # Total sum and mean of collater_valueofguarantee_1124L
                pl.col("collater_valueofguarantee_1124L").cast(pl.Float64).sum().alias("collater_valueofguarantee_1124L_sum"),
                pl.col("collater_valueofguarantee_1124L").cast(pl.Float64).mean().alias("collater_valueofguarantee_1124L_mean"),

                # Number of non-null collater_valueofguarantee_876L
                pl.when(pl.col("collater_valueofguarantee_876L").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collater_valueofguarantee_876L"),
                # Total sum and mean of collater_valueofguarantee_876L
                pl.col("collater_valueofguarantee_876L").cast(pl.Float64).sum().alias("collater_valueofguarantee_876L_sum"),
                pl.col("collater_valueofguarantee_876L").cast(pl.Float64).mean().alias("collater_valueofguarantee_876L_mean"),

                # Number of non-null collaterals_typeofguarante_359M
                pl.when(pl.col("collaterals_typeofguarante_359M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collaterals_typeofguarante_359M"),
                # Number of non-null collaterals_typeofguarante_669M
                pl.when(pl.col("collaterals_typeofguarante_669M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_collaterals_typeofguarante_669M"),

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
                    ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_overdue_1140A"),
                pl.col("pmts_overdue_1140A").sum().alias("pmts_overdue_1140A_sum"),
                pl.col("pmts_overdue_1140A").mean().alias("pmts_overdue_1140A_mean"),
                pl.col("pmts_overdue_1140A").filter(
                            (pl.col("pmts_overdue_1140A").is_not_null()) & (pl.col("pmts_overdue_1140A").gt(0.0))
                    ).last().alias("pmts_overdue_1140A_last"),

                pl.when(
                            (pl.col("pmts_overdue_1152A").is_not_null()) & (pl.col("pmts_overdue_1152A").gt(0.0))
                    ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_overdue_1152A"),
                pl.col("pmts_overdue_1152A").sum().alias("pmts_overdue_1152A_sum"),
                pl.col("pmts_overdue_1152A").mean().alias("pmts_overdue_1152A_mean"),
                pl.col("pmts_overdue_1152A").filter(
                            (pl.col("pmts_overdue_1152A").is_not_null()) & (pl.col("pmts_overdue_1152A").gt(0.0))
                    ).last().alias("pmts_overdue_1152A_last"),

                # Number of non-null subjectroles_name_541M
                pl.when(pl.col("subjectroles_name_541M").replace("a55475b1", None).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_subjectroles_name_541M"),

                # Years of payments of closed credit (pmts_year_507T) contract and the current contract (pmts_year_1139T)
                # Number of non-null pmts_year_507T
                pl.when(pl.col("pmts_year_507T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_year_507T"),
                # First and last years of payments of closed credit (pmts_year_507T) contract, avoiding null, and their difference
                pl.col("pmts_year_507T").cast(pl.Float64).min().alias("pmts_year_507T_first"),
                pl.col("pmts_year_507T").cast(pl.Float64).max().alias("pmts_year_507T_last"),
                (pl.col("pmts_year_507T").cast(pl.Float64).max() - pl.col("pmts_year_507T").cast(pl.Float64).min()).alias("pmts_year_507T_duration"),
                # Number of non-null pmts_year_507T
                pl.when(pl.col("pmts_year_1139T").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_year_1139T"),
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
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],

                # Pick last three values of groupby and compute mean for all numerical columns
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

            # Dropped completely: pmts_month_158T, pmts_month_706T
            # Implemented: pmts_year_1139T, pmts_year_507T
            
        elif table_name=='credit_bureau_b_2':
            # Fill null for pmts_dpdvalue_108P (observed)
            #data = self.encode_categorical_columns(data, table_name)
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_group2', '_id']

            data = data.group_by(['case_id', 'num_group1']).agg(
                # Number of non-null pmts_date_1107D (it has type pl.Date)
                pl.when(pl.col("pmts_date_1107D").is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmts_date_1107D"),
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
                    ).arg_max().alias("pmts_dpdvalue_108P_maxidx"),
                # TODO: check here which positive trend is better
                #(pl.col("pmts_dpdvalue_108P").max() - pl.col("pmts_dpdvalue_108P").last()).alias("pmts_dpdvalue_108P_pos"),
                ((pl.col("pmts_dpdvalue_108P").max() - pl.col("pmts_dpdvalue_108P").last())/pl.col("pmts_dpdvalue_108P").max()).fill_nan(1.0).alias("pmts_dpdvalue_108P_pos"),

                # pmts_pmtsoverdue_635A values
                pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
                pl.col("pmts_pmtsoverdue_635A").last().alias("pmts_pmtsoverdue_635A_last"),
                pl.col("pmts_pmtsoverdue_635A").filter(
                        (pl.col("pmts_pmtsoverdue_635A").is_not_null()) & (pl.col("pmts_pmtsoverdue_635A").gt(0.0))
                    ).arg_max().alias("pmts_pmtsoverdue_635A_maxidx"),
                ((pl.col("pmts_pmtsoverdue_635A").max() - pl.col("pmts_pmtsoverdue_635A").last())/pl.col("pmts_pmtsoverdue_635A").max()).fill_nan(1.0).alias("pmts_pmtsoverdue_635A_pos"),
        
                # Pick last three values of groupby and compute mean for all numerical columns
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        return data
    
    def aggregate_depth_1(self, data: pl.DataFrame, table_name: str) -> pl.DataFrame:
        """
        Aggregate data by case_id
        """
        if table_name=='credit_bureau_b_1':
            # Encoding categorical columns
            #data = self.encode_categorical_columns(data, table_name)
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']

            # Columns to comute Summary Statistics (max, sum, mean, median)
            summary_columns = ['amount_1115A', 'totalamount_503A', 'totalamount_881A', 'overdueamountmax_950A',
                               'interesteffectiverate_369L','interestrateyearly_538L', 'pmtdaysoverdue_1135P', 'pmtnumpending_403L']
            mean_columns = ['installmentamount_644A', 'installmentamount_833A', 'residualamount_1093A', 'residualamount_3940956A',
                            'instlamount_892A','debtvalue_227A','debtpastduevalue_732A',
                            'dpd_550P','dpd_733P','dpdmax_851P', 'residualamount_127A','numberofinstls_810L',
                            # more added
                            'credlmt_1052A','credlmt_228A','credlmt_3940954A','credquantity_1099L','credquantity_984L']
            sum_columns = ['installmentamount_833A', 'residualamount_3940956A', 'credlmt_1052A','credlmt_228A','credlmt_3940954A','credquantity_1099L',
                           'credquantity_984L','instlamount_892A','debtvalue_227A','debtpastduevalue_732A', 'residualamount_127A','numberofinstls_810L']
            max_columns = ['credquantity_1099L', 'credquantity_984L', 'maxdebtpduevalodued_3940955A', 'dpd_550P','dpd_733P','dpdmax_851P']
            min_columns = ['maxdebtpduevalodued_3940955A']
            year_columns = ['overdueamountmaxdateyear_432T']

            # Similar lists for depth_2 table
            summary_columns += ["pmts_dpdvalue_108P_max","pmts_dpdvalue_108P_last","pmts_pmtsoverdue_635A_max","pmts_pmtsoverdue_635A_last"]
            mean_columns += ["pmts_date_1107D_duration","pmts_date_1107D_duration_days","pmts_dpdvalue_108P_pos","pmts_pmtsoverdue_635A_pos"]
            sum_columns += ["num_pmts_date_1107D","num_pmts_dpdvalue_108P"]
            max_columns += ["pmts_date_1107D_duration","pmts_date_1107D_duration_days","pmts_dpdvalue_108P_maxidx","pmts_pmtsoverdue_635A_maxidx"]
            min_columns += []

            # Aggregating by case_id
            data = data.group_by('case_id').agg(

                # Number of non-null entries in summary columns
                *[pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_"+col) for col in summary_columns],

                # Create new features from summary columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in summary_columns],
                
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in summary_columns],

                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in summary_columns],

                *[pl.col(col).cast(pl.Float64).filter(
                       (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                       ).median().alias(col+"_median") for col in summary_columns],

                # Create mean values for columns in mean_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in mean_columns],

                # Create columns with sum values from sum_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in sum_columns],

                # Create columns with max values from max_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in max_columns],

                # Create columns with min values from min_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).min().alias(col+"_min") for col in min_columns],


                # A binary feature indicating whether a column has at least one missing value (null)
                # TODO: cast bool to int16?
                (pl.col("amount_1115A").is_null()).any().cast(pl.Int16).alias("amount_1115A_null"),
                (pl.col("totalamount_881A").is_null()).any().cast(pl.Int16).alias("totalamount_881A_null"),
                
                # Credit Utilization Ratio: ratio of amount_1115A to totalamount_503A 
                (pl.col("amount_1115A") / pl.col("totalamount_503A")).fill_null(0.0).replace(float("inf"), None).max().alias("amount_1115A_totalamount_503A_ratio_max"),
                (pl.col("amount_1115A").max() / pl.col("credlmt_3940954A").fill_null(0.0).max()).replace(float("inf"), None).fill_nan(None).alias("amount_1115A_credlmt_3940954A_ratio"),
                (pl.col("amount_1115A") / pl.col("credlmt_3940954A").fill_null(0.0)).replace(float("inf"),None).fill_nan(None).max().alias("amount_1115A_credlmt_3940954A_ratio_max"),
                (pl.col("amount_1115A").max() / pl.col("credlmt_1052A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("amount_1115A_credlmt_1052A_ratio"),
                (pl.col("amount_1115A") / pl.col("credlmt_1052A").fill_null(0.0)).replace(float("inf"),None).fill_nan(None).max().alias("amount_1115A_credlmt_1052A_ratio_max"),
        

                # Compute the absolute difference between totalamount_503A and amount_1115A.
                (pl.col("totalamount_503A") - pl.col("amount_1115A")).abs().max().alias("totalamount_503A_max_diff"),
                # Difference in Credit Limits
                (pl.col("credlmt_3940954A") - pl.col("credlmt_228A")).max().alias("credlmt_3940954A_credlmt_228A_diff_max"),
                (pl.col("credlmt_3940954A") - pl.col("credlmt_1052A")).max().alias("credlmt_3940954A_credlmt_1052A_diff_max"),
                (pl.col("credlmt_3940954A").max() - pl.col("credlmt_228A").max()).alias("credlmt_3940954A_credlmt_228A_diff"),
                (pl.col("credlmt_3940954A").max() - pl.col("credlmt_1052A").max()).alias("credlmt_3940954A_credlmt_1052A_diff"),

                # TODO: Log Transformations: If the credit amounts have a wide range, consider taking the logarithm to normalize the distribution.

                # Max value of totalamount_503A over totalamount_881A
                (pl.col("totalamount_503A").max() / pl.col("totalamount_881A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("totalamount_503A_881A"),

                ## Columns with years
                #*[pl.col(col).cast(pl.Float64).max().fill_null(0.0).alias(col+"_last") for col in year_columns],
                #*[(pl.col(col).cast(pl.Float64).max() - pl.col(col).cast(pl.Float64).min()).fill_null(0.0).alias(col+"_duration") for col in year_columns],

                # Overdue-to-Installment Ratio
                (pl.col("overdueamountmax_950A").max() / pl.col("installmentamount_833A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("overdueamountmax_950A_installmentamount_833A_ratio"),
                (pl.col("overdueamountmax_950A").max() / pl.col("instlamount_892A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("overdueamountmax_950A_instlamount_892A_ratio"),
                
                # Residual-to-Credit Limit Ratio
                (pl.col("residualamount_3940956A").max() / pl.col("credlmt_3940954A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("residualamount_3940956A_credlmt_3940954A_ratio"),
                (pl.col("residualamount_3940956A") / pl.col("credlmt_3940954A").fill_null(0.0)).replace(float("inf"),None).fill_nan(None).max().alias("residualamount_3940956A_credlmt_3940954A_ratio_max"),
        
                # Create a binary feature indicating whether the maximum debt occurred recently (e.g., within the last month or quarter) based on maxdebtpduevalodued_3940955A
                (pl.col("maxdebtpduevalodued_3940955A").filter(
                        (pl.col("maxdebtpduevalodued_3940955A").is_not_null()) & (pl.col("maxdebtpduevalodued_3940955A").gt(0.0))
                    ).min() < 120.0).cast(pl.Int16).alias("maxdebtpduevalodued_3940955A_isrecent"),

                # TODO Debt-to-Income Ratio: Divide the outstanding debt amount (debtvalue_227A) by the client’s income (if available). This ratio provides insights into a client’s ability to manage debt relative to their income.

                # Past Due Ratio: Calculate the ratio of unpaid debt (debtpastduevalue_732A) to the total outstanding debt (debtvalue_227A). High past due ratios may indicate higher credit risk.
                (pl.col("debtpastduevalue_732A").max() / pl.col("debtvalue_227A").fill_null(0.0).max()).replace(float("inf"),None).fill_nan(None).alias("debtpastduevalue_732A_debtvalue_227A_ratio"),
                (pl.col("debtpastduevalue_732A") / pl.col("debtvalue_227A").fill_null(0.0)).replace(float("inf"), None).fill_nan(None).max().alias("debtpastduevalue_732A_debtvalue_227A_ratio_max"),


                # Number of non-null and greater than 0.0 values in dpd_550P, dpd_733P and dpdmax_851P columns
                pl.when( (pl.col("dpd_550P").is_not_null()) & (pl.col("dpd_550P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_dpd_550P"),
                pl.when( (pl.col("dpd_733P").is_not_null()) & (pl.col("dpd_733P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_dpd_733P"),
                pl.when( (pl.col("dpdmax_851P").is_not_null()) & (pl.col("dpdmax_851P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_dpdmax_851P"),
                

                # Columns for month (dpdmaxdatemonth_804T) and years (dpdmaxdateyear_742T)
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max() - 
                    pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).min()).dt.total_days().alias("dpdmaxdate_duration"),
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max()).dt.year().alias("dpdmaxdateyear_742T_last"),
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max()).dt.month().alias("dpdmaxdatemonth_804T_last"),

                # Columns for month (overdueamountmaxdatemonth_494T) and years (overdueamountmaxdateyear_432T)
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max() - 
                    pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).min()).dt.total_days().alias("overdueamountmaxdat_duration"),
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max()).dt.year().alias("overdueamountmaxdateyear_432T_last"),
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max()).dt.month().alias("overdueamountmaxdatemonth_494T_last"),

                # Durations: last update - contract date
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().min().alias('lastupdate_260D_contractdate_551D_diff_min'),
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().max().alias('lastupdate_260D_contractdate_551D_diff_max'),
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().mean().alias('lastupdate_260D_contractdate_551D_diff_mean'),
                # Duration:  contract maturity date - last update
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().min().alias('contractmaturitydate_151D_lastupdate_260D_diff_min'),
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().max().alias('contractmaturitydate_151D_lastupdate_260D_diff_max'),
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().mean().alias('contractmaturitydate_151D_lastupdate_260D_diff_mean'),

                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                #*[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Interval columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_interval")],
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],


                # Columns from credit_bureau_b_2
                (pl.col("pmts_date_1107D_last").max() - pl.col("pmts_date_1107D_first").min()).alias("pmts_date_1107D_duration"),
                (pl.col("pmts_date_1107D_last").max() - pl.col("pmts_date_1107D_first").min()).mul(365).alias("pmts_date_1107D_duration_days"),
                pl.col("pmts_date_1107D_first").min().alias("pmts_date_1107D_first"),
                pl.col("pmts_date_1107D_last").max().alias("pmts_date_1107D_last"),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        if table_name=='credit_bureau_a_1':

            # TODO
            # - 'annualeffectiverate_199L','annualeffectiverate_63L': consider aggregating by other relevant features, such as 
            #    1) loan type,
            #    2) borrower credit score, 
            #    3) loan purpose (purposeofcred_426M,purposeofcred_722M,purposeofcred_874M).
            #   This way, you can analyze how interest rates vary across different segments.

            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']


            # Columns to comute Summary Statistics (max, sum, mean, median)
            summary_columns = ['annualeffectiverate_199L','annualeffectiverate_63L','contractsum_5085717L','interestrate_508L']
            mean_columns = ['credlmt_230A','credlmt_935A','nominalrate_281L','nominalrate_498L','numberofinstls_229L','numberofinstls_320L','numberofoutstandinstls_520L',
                            'numberofoutstandinstls_59L',
                            'numberofoverdueinstls_725L','numberofoverdueinstls_834L','periodicityofpmts_1102L','periodicityofpmts_837L','prolongationcount_1120L',
                            'prolongationcount_599L',
                            'totalamount_6A','totalamount_996A','numberofoverdueinstlmax_1039L','residualamount_856A',
                            # more columns
                            'debtoutstand_525A','debtoverdue_47A','dpdmax_139P','dpdmax_757P','instlamount_852A','instlamount_768A',
                            'monthlyinstlamount_674A','monthlyinstlamount_332A','numberofcontrsvalue_258L','numberofcontrsvalue_358L',
                            'outstandingamount_354A','outstandingamount_362A','overdueamount_31A','overdueamount_659A','overdueamountmax_155A','overdueamountmax_35A',
                            'residualamount_488A','totaldebtoverduevalue_178A','totaldebtoverduevalue_718A','totaloutstanddebtvalue_39A','totaloutstanddebtvalue_668A']
            sum_columns = ['credlmt_230A','credlmt_935A','debtoutstand_525A','debtoverdue_47A','dpdmax_139P','dpdmax_757P','instlamount_852A','instlamount_768A',
                           'monthlyinstlamount_674A','monthlyinstlamount_332A','numberofcontrsvalue_258L','numberofcontrsvalue_358L','numberofinstls_229L','numberofinstls_320L',
                           'numberofoutstandinstls_520L','numberofoutstandinstls_59L','numberofoverdueinstlmax_1039L','outstandingamount_354A','outstandingamount_362A',
                           'overdueamount_31A','overdueamount_659A','overdueamountmax_155A','overdueamountmax_35A','prolongationcount_1120L','prolongationcount_599L',
                           'residualamount_488A','totalamount_6A','totalamount_996A','totaldebtoverduevalue_178A','totaldebtoverduevalue_718A',
                           'totaloutstanddebtvalue_39A','totaloutstanddebtvalue_668A']
                            # TODO in sum: 'numberofoverdueinstlmax_1039L','numberofoverdueinstlmax_1151L','numberofoverdueinstls_725L','numberofoverdueinstls_834L'
                            # Removed: residualamount_856A
            max_columns = ['credlmt_230A','credlmt_935A','dpdmax_139P','dpdmax_757P','overdueamountmax2_14A','overdueamountmax2_398A','prolongationcount_1120L','prolongationcount_599L',
                           'outstandingamount_362A']
            min_columns = []
            std_columns = ['nominalrate_281L','nominalrate_498L','residualamount_856A','residualamount_488A']
            number_non0s_column = ['dpdmax_139P','dpdmax_757P','monthlyinstlamount_674A','monthlyinstlamount_332A','numberofoutstandinstls_520L','numberofoutstandinstls_59L',
                                   'numberofoverdueinstlmax_1151L','numberofoverdueinstls_725L','numberofoverdueinstls_834L','dateofcredend_353D','dateofcredend_289D']

            # Similar lists for depth_2 table
            summary_columns += []
            mean_columns += []
            sum_columns += []
            max_columns += []
            min_columns += []
            std_columns += []

            collater_typofvalofguarant_unique = ['9a0c095e','8fd95e4b','06fb9ba8','3cbe86ba']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Number of non-null entries in summary columns
                *[pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_"+col) for col in summary_columns],

                # Number of non-null entries and non-zeros in number_non0s_column columns
                *[pl.when(
                    (pl.col(col).is_not_null()) & (pl.col(col).cast(pl.Float64).gt(0.0))
                    ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_"+col) for col in number_non0s_column],

                # Create new features from summary columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in summary_columns],
                
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in summary_columns],

                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in summary_columns],

                # Create mean values for columns in mean_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in mean_columns],

                # Create std values for columns in std_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).std().alias(col+"_std") for col in std_columns],

                # Create columns with sum values from sum_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in sum_columns],

                # Create columns with max values from max_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in max_columns],

                # Create columns with min values from min_columns
                *[pl.col(col).cast(pl.Float64).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).min().alias(col+"_min") for col in min_columns],

                
                # Diffs
                (pl.col('annualeffectiverate_63L').cast(pl.Float64,strict=False).sum() - pl.col('annualeffectiverate_199L').cast(pl.Float64,strict=False).sum()).alias('annualeffectiverate_63L_199L_sum_diff'),
                (pl.col('annualeffectiverate_63L').cast(pl.Float64,strict=False).mean() - pl.col('annualeffectiverate_199L').cast(pl.Float64,strict=False).mean()).alias('annualeffectiverate_63L_199L_mean_diff'),

                (pl.col('credlmt_935A').sum() - pl.col('credlmt_230A').sum()).alias('credlmt_935A_1230A_sum_diff'),
                (pl.col('credlmt_935A').mean() - pl.col('credlmt_230A').mean()).alias('credlmt_935A_230A_mean_diff'),

                # Interest Rate Spread: Calculate the difference between the nominal interest rates for active and closed contracts. This spread could be indicative of risk.
                (pl.col('nominalrate_281L').cast(pl.Float64,strict=False).mean() - pl.col('nominalrate_498L').cast(pl.Float64,strict=False).mean()).alias('nominalrate_281L_498L_mean_diff'),

                # Contract Sum Spread: Calculate the difference between the sum of active and closed contracts. This spread could be indicative of risk.
                (pl.col('contractsum_5085717L').cast(pl.Float64,strict=False).mean() - pl.col('contractsum_5085717L').cast(pl.Float64,strict=False).mean()).alias('contractsum_5085717L_mean_diff'),

                # DPD Spread: Calculate the difference between the maximum DPD values for active and closed contracts. This spread could be indicative of risk.
                (pl.col('dpdmax_139P').mean() - pl.col('dpdmax_757P').mean()).alias('dpdmax_139P_757P_mean_diff'),

                # Instalment Difference
                (pl.col('instlamount_768A').sum() - pl.col('instlamount_852A').sum()).alias('instlamount_768A_852A_diff'),

                # Overdue Percentage: Calculate the percentage of overdue debt (debtoverdue_47A) relative to the total outstanding debt (debtoutstand_525A). High percentages may signal credit risk.
                (pl.col('debtoverdue_47A').sum() / pl.col('debtoutstand_525A').sum().fill_null(0.0)).replace(float("inf"),None).fill_nan(None).alias('debtoverdue_47A_debtoutstand_525A_ratio'),
                
                # Debt Utilization: Divide the outstanding debt by the credit limit. High utilization ratios may indicate risk.
                (pl.col('debtoutstand_525A').sum() / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('debtoutstand_525A_credlmt_935A_ratio'),

                # Instalment Coverage Ratio: Divide the total amount paid (instlamount_852A) by the total amount due (instlamount_768A). A higher ratio suggests better payment behavior.
                (pl.col('instlamount_852A').sum() / pl.col('instlamount_768A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('instlamount_852A_768A_ratio'),

                # Instalment Difference: Calculate the difference between monthlyinstlamount_674A and monthlyinstlamount_332A. A larger difference could be relevant for risk assessment.
                (pl.col('monthlyinstlamount_332A').sum() - pl.col('monthlyinstlamount_674A').sum()).alias('monthlyinstlamount_332A_674A_diff'),

                # Instalment Coverage Ratio: Divide the total amount paid (monthlyinstlamount_674A) by the total amount due (monthlyinstlamount_332A). A higher ratio suggests better payment behavior.
                (pl.col('monthlyinstlamount_332A').sum() / pl.col('monthlyinstlamount_674A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('monthlyinstlamount_332A_674A_ratio'),

                # Instalment Stability: Compare the number of instalments for active contracts with the average number of instalments for closed contracts. A significant deviation might indicate instability.
                (pl.col('numberofinstls_320L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_229L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"),None).fill_nan(None).alias('numberofinstls_320L_229L_sum_ratio'),
                (pl.col('numberofinstls_320L').cast(pl.Float64,strict=False).mean() / pl.col('numberofinstls_229L').cast(pl.Float64,strict=False).mean().fill_null(0.0)).replace(float("inf"),None).fill_nan(None).alias('numberofinstls_320L_229L_mean_ratio'),

                # the ratio of the actual number of outstanding instalments 'numberofoutstandinstls_59L' to the total number of instalments 'numberofinstls_320L' for active contracts.
                (pl.col('numberofoutstandinstls_59L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_320L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoutstandinstls_59L_numberofinstls_320L_ratio'),
                # the ratio of the actual number of outstanding instalments 'numberofoutstandinstls_520L' to the total number of instalments 'numberofinstls_229L' for closed contracts.
                (pl.col('numberofoutstandinstls_520L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_229L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoutstandinstls_520L_numberofinstls_229L_ratio'),

                # Ratio of numberofoverdueinstlmax_1039L to numberofinstls_320L for active contract
                (pl.col('numberofoverdueinstlmax_1039L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_320L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoverdueinstlmax_1039L_numberofinstls_320L_ratio'),
                # Ratio of numberofoverdueinstlmax_1039L to numberofinstls_229L for closed contract
                (pl.col('numberofoverdueinstlmax_1151L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_229L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoverdueinstlmax_1151L_numberofinstls_229L_ratio'),

                # the ratio of the actual number of overdue instalments 'numberofoverdueinstls_725L' to the total number of instalments 'numberofinstls_320L' for active contracts.
                (pl.col('numberofoverdueinstls_725L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_320L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoverdueinstls_725L_numberofinstls_320L_ratio'),
                # the ratio of the actual number of overdue instalments 'numberofoverdueinstls_834L' to the total number of instalments 'numberofinstls_229L' for closed contracts.
                (pl.col('numberofoverdueinstls_834L').cast(pl.Float64,strict=False).sum() / pl.col('numberofinstls_229L').cast(pl.Float64,strict=False).sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('numberofoverdueinstls_834L_numberofinstls_229L_ratio'),

                # Ratio of outstanding amount outstandingamount_354A to credit limit credlmt_230A for closed contracts
                (pl.col('outstandingamount_354A').sum() / pl.col('credlmt_230A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('outstandingamount_354A_credlmt_230A_ratio'),
                # Ratio of outstanding amount outstandingamount_362A to credit limit credlmt_935A for active contracts
                (pl.col('outstandingamount_362A').sum() / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('outstandingamount_362A_credlmt_935A_ratio'),

                # Ratio of overdue amount overdueamount_31A to outstanding amount outstandingamount_354A for closed contracts
                (pl.col('overdueamount_31A').sum() / pl.col('outstandingamount_354A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('overdueamount_31A_outstandingamount_354A_ratio'),
                # Ratio of overdue amount overdueamount_659A to outstanding amount outstandingamount_362A for active contracts
                (pl.col('overdueamount_659A').sum() / pl.col('outstandingamount_362A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('overdueamount_659A_outstandingamount_362A_ratio'),

                # Residual Ratio: Compute the ratio between residualamount_856A and residualamount_488A.
                (pl.col('residualamount_856A').sum() / pl.col('residualamount_488A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('residualamount_856A_488A_ratio'),
                # Normalized Residual Amounts: Calculate the normalized residual amounts by dividing each residual amount by the credit limit
                (pl.col('residualamount_856A').sum() / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('residualamount_856A_credlmt_935A_ratio'),
                (pl.col('residualamount_488A').sum() / pl.col('credlmt_230A').sum().fill_null(0.0)).replace(float("inf"), None).fill_nan(None).alias('residualamount_488A_credlmt_230A_ratio'),

                # Contract Status Proportion: Calculate the proportion of active contracts (totalamount_996A) relative to the total (totalamount_996A + totalamount_6A) contracts.
                (pl.col('totalamount_6A').sum() / (pl.col('totalamount_996A').sum().fill_null(0.0) + pl.col('totalamount_6A').sum().fill_null(0.0))).replace(float("inf"), None).fill_nan(None).alias('totalamount_6A_totalamount_996A_ratio'),


                # Durations: mean, max,min value of durations of closed contracts ('dateofcredend_353D','dateofcredstart_181D')
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().mean().alias('dateofcredend_353D_dateofcredstart_181D_diff_mean'),
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().max().alias('dateofcredend_353D_dateofcredstart_181D_diff_max'),
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().min().alias('dateofcredend_353D_dateofcredstart_181D_diff_min'),

                # Durations: mean, max,min value of durations of active contracts ('dateofcredend_289D','dateofcredstart_739D')
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().mean().alias('dateofcredend_289D_dateofcredstart_739D_diff_mean'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().max().alias('dateofcredend_289D_dateofcredstart_739D_diff_max'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().min().alias('dateofcredend_289D_dateofcredstart_739D_diff_min'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().std().alias('dateofcredend_289D_dateofcredstart_739D_diff_std'),

                # Difference between dateofcredend_353D and dateofrealrepmt_138D
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().mean().alias('dateofcredend_353D_dateofrealrepmt_138D_diff'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().max().alias('dateofcredend_353D_dateofrealrepmt_138D_max'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().min().alias('dateofcredend_353D_dateofrealrepmt_138D_min'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().std().alias('dateofcredend_353D_dateofrealrepmt_138D_std'),

                # Last updates:
                #pl.col('lastupdate_1112D').max().alias('lastupdate_1112D_max'),
                #pl.col('lastupdate_388D').max().alias('lastupdate_388D_max'),
                #pl.col('lastupdate_1112D').min().alias('lastupdate_1112D_min'), # Contracts without long time update?
                #pl.col('lastupdate_388D').min().alias('lastupdate_388D_min'),   # Contracts without long time update?
                pl.col('dateofcredstart_739D').drop_nulls().mean().alias('dateofcredstart_739D_mean'),
                pl.col('dateofcredstart_181D').drop_nulls().mean().alias('dateofcredstart_181D_mean'),
                pl.col('dateofrealrepmt_138D').drop_nulls().mean().alias('dateofrealrepmt_138D_mean'),
                pl.col('dateofcredend_353D').drop_nulls().mean().alias('dateofcredend_353D_mean'),
                pl.col('dateofcredend_353D').drop_nulls().max().alias('dateofcredend_353D_max'),
                 pl.col('dateofcredstart_181D').drop_nulls().max().alias('dateofcredstart_181D_max'),


                # Latest date with maximum number of overdue instl (numberofoverdueinstlmaxdat_148D) and (numberofoverdueinstlmaxdat_641D)
                #pl.col('numberofoverdueinstlmaxdat_148D').max().alias('numberofoverdueinstlmaxdat_148D_max'),
                #pl.col('numberofoverdueinstlmaxdat_641D').max().alias('numberofoverdueinstlmaxdat_641D_max'),
                (pl.col('refreshdate_3813885D').max() - pl.col('numberofoverdueinstlmaxdat_148D').max()).dt.total_days().alias('refreshdate_3813885D_numberofoverdueinstlmaxdat_148D_diff'),
                (pl.col('refreshdate_3813885D').max() - pl.col('numberofoverdueinstlmaxdat_641D').max()).dt.total_days().alias('refreshdate_3813885D_numberofoverdueinstlmaxdat_641D_diff'),

                # remaining time of max overdue installments date till contract end
                (pl.col("dateofcredend_353D") - pl.col("numberofoverdueinstlmaxdat_148D")).dt.total_days().min().alias('dateofcredend_353D_numberofoverdueinstlmaxdat_148D_diff'),
                (pl.col("dateofcredend_289D") - pl.col("numberofoverdueinstlmaxdat_641D")).dt.total_days().min().alias('dateofcredend_289D_numberofoverdueinstlmaxdat_641D_diff'),

                # Latest date with maximal overdue amount (overdueamountmax2date_1002D) and (overdueamountmax2date_1142D)
                #pl.col('overdueamountmax2date_1002D').max().alias('overdueamountmax2date_1002D_max'),
                #pl.col('overdueamountmax2date_1142D').max().alias('overdueamountmax2date_1142D_max'),
                (pl.col('refreshdate_3813885D').max() - pl.col('overdueamountmax2date_1002D').max()).dt.total_days().alias('refreshdate_3813885D_overdueamountmax2date_1002D_diff'),
                (pl.col('refreshdate_3813885D').max() - pl.col('overdueamountmax2date_1142D').max()).dt.total_days().alias('refreshdate_3813885D_overdueamountmax2date_1142D_diff'),
                
                # remaining time of max overdue amount date till contract end
                (pl.col("dateofcredend_353D") - pl.col("overdueamountmax2date_1002D")).dt.total_days().min().alias('dateofcredend_353D_overdueamountmax2date_1002D_diff'),
                (pl.col("dateofcredend_289D") - pl.col("overdueamountmax2date_1142D")).dt.total_days().min().alias('dateofcredend_289D_overdueamountmax2date_1142D_diff'),

                # Date
                pl.col('overdueamountmax2date_1142D').max().alias('overdueamountmax2date_1142D_max'),
                pl.col('numberofoverdueinstlmaxdat_148D').max().alias('numberofoverdueinstlmaxdat_148D_max'),
                pl.col('numberofoverdueinstlmaxdat_641D').drop_nulls().mean().alias('numberofoverdueinstlmaxdat_641D_mean'),
                pl.col('overdueamountmax2date_1002D').drop_nulls().mean().alias('overdueamountmax2date_1002D_mean'),
                pl.col('dateofcredend_289D').drop_nulls().mean().alias('dateofcredend_289D_mean'),
                
                

                # Date from str
                #(pl.date(pl.col("overdueamountmaxdateyear_994T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_284T").cast(pl.Float64), 1).max()).alias("overdueamountmaxdate_994T_284T_fromstr_last"),
                #(pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1).max()).alias("overdueamountmaxdate_2T_365T_fromstr_last"),

                # remaining time of max overdue amount date till contract end (from str version)
                (pl.col("dateofcredend_353D") - pl.date(pl.col("overdueamountmaxdateyear_994T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_284T").cast(pl.Float64), 1)).dt.total_days().min().alias('dateofcredend_353D_overdueamountmaxdate_994T_284T_diff'),
                (pl.col("dateofcredend_289D") - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().min().alias('dateofcredend_289D_overdueamountmaxdate_2T_365T_diff'),
                # Assuming refreshdate_3813885D is current date, computing how much time ago this max overdue amount happend
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().min().alias("overdueamountmaxdate_2T_365T_refreshed_last"),
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().mean().alias("overdueamountmaxdate_2T_365T_refreshed_mean"),

                # Date from str
                #(pl.date(pl.col("dpdmaxdateyear_896T").cast(pl.Float64),pl.col("dpdmaxdatemonth_442T").cast(pl.Float64), 1).max()).alias("dpdmaxdate_896T_442T_fromstr_last"),
                #(pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1).max()).alias("dpdmaxdate_596T_89T_fromstr_last"),

                 # remaining time of max dpd date till contract end (from str version)
                (pl.col("dateofcredend_353D") - pl.date(pl.col("dpdmaxdateyear_896T").cast(pl.Float64),pl.col("dpdmaxdatemonth_442T").cast(pl.Float64), 1)).dt.total_days().min().alias('dateofcredend_353D_dpdmaxdate_896T_442T_diff'),
                (pl.col("dateofcredend_289D") - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().min().alias('dateofcredend_289D_dpdmaxdate_596T_89T_diff'),
                # Assuming refreshdate_3813885D is current date, computing how much time ago this max overdue amount happend
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().min().alias("dpdmaxdateyear_596T_89T_refreshed_last"),
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().mean().alias("dpdmaxdateyear_596T_89T_refreshed_mean"),

                # Refresh date info
                #pl.col('refreshdate_3813885D').min().alias('refreshdate_3813885D_min'),
                pl.col('refreshdate_3813885D').max().alias('refreshdate_3813885D_max'),
                #pl.col('refreshdate_3813885D').mean().alias('refreshdate_3813885D_mean'),
                # difference between max and min values of refreshdate_3813885D in days
                (pl.col('refreshdate_3813885D').max() - pl.col('refreshdate_3813885D').min()).dt.total_days().alias('refreshdate_3813885D_diff'),
                # standard deviation of refreshdate_3813885D in days
                (pl.col('refreshdate_3813885D') - pl.col('refreshdate_3813885D').mean()).std().dt.total_days().alias('refreshdate_3813885D_std'),



                # Difference with respect to refresh date
                (pl.col('dateofcredend_289D') - pl.col('refreshdate_3813885D')).dt.total_days().mean().alias('dateofcredend_289D_refreshdate_3813885D_diff_mean'),
                (pl.col('dateofcredend_289D') - pl.col('refreshdate_3813885D')).dt.total_days().std().alias('dateofcredend_289D_refreshdate_3813885D_diff_std'),
                # ratios of durations
                ((pl.col('dateofcredend_289D') - pl.col('refreshdate_3813885D')).dt.total_days().mean() / (pl.col('dateofcredend_289D') - pl.col('dateofcredstart_739D')).dt.total_days().mean()).alias('dateofcredend_289D_refreshdate_3813885D_diff_mean_norm'),
                ((pl.col('dateofcredend_289D') - pl.col('refreshdate_3813885D')).dt.total_days().std() / (pl.col('dateofcredend_289D') - pl.col('dateofcredstart_739D')).dt.total_days().mean()).alias('dateofcredend_289D_refreshdate_3813885D_diff_std_norm'),


                # Calculate the duration (in days) between the current date and the lastupdate_1112D (for active contracts) or lastupdate_388D (for closed contracts).
                (pl.col('refreshdate_3813885D') - pl.col('lastupdate_1112D')).dt.total_days().mean().alias('refreshdate_3813885D_lastupdate_1112D_diff_mean'),
                (pl.col('refreshdate_3813885D') - pl.col('lastupdate_388D')).dt.total_days().mean().alias('refreshdate_3813885D_lastupdate_388D_diff_mean'),
                (pl.col('refreshdate_3813885D') - pl.col('lastupdate_388D')).dt.total_days().min().alias('refreshdate_3813885D_lastupdate_388D_diff_min'),

                ######################################
                ##### Depth 2 data --->

                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                #*[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_last")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_last")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_mode")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_mode")],

                # TODO: check more depth 2 aggregations
                # Aggregate with sum if column starts with 'num_', but ignore the column 'num_group1'
                *[pl.col(col).sum().alias(col) for col in data.columns if col.startswith("num_") and col != "num_group1"],
                #*[pl.col(col).sum().alias(col+'_sum') for col in data.columns if col.startswith("num_")],
                # Aggregate with mean if column ends with '_mean'
                *[pl.col(col).mean().alias(col+'_mean') for col in data.columns if col.endswith("_mean")],
                # Aggregare with min if columns ends with '_first'
                *[pl.col(col).min().alias(col+'_min') for col in data.columns if col.endswith("_first")],
                # Aggregate with max if columns ends with '_last'
                *[pl.col(col).max().alias(col+'_max') for col in data.columns if col.endswith("_last")],
                # Aggregate with mean if columns ends with '_duration'
                *[pl.col(col).mean().alias(col+'_mean') for col in data.columns if col.endswith("_duration")],
                # Aggregate with mean if column ends with '_diff'
                *[pl.col(col).mean().alias(col+'_mean') for col in data.columns if col.endswith("_diff")],
                
                ###### Depth 2 one-hot-encoded features
                # Sum of one-hot-encoded columns
                #*[pl.col(f"collater_typofvalofguarant_{role}_298M").sum().alias(f"collater_typofvalofguarant_{role}_298M") for role in collater_typofvalofguarant_unique],
                #*[pl.col(f"collater_typofvalofguarant_{role}_407M").sum().alias(f"collater_typofvalofguarant_{role}_407M") for role in collater_typofvalofguarant_unique],

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        if table_name=='person_1':

            # One-hot-encoded features
            persontype_1072L_unique = ['1.0', '4.0', '5.0']
            education_927M_unique = ['P17_36_170', 'a55475b1', 'P106_81_188', 'P97_36_170', 'P157_18_172', 'P33_146_175']
            empl_employedtotal_800L_unique = ['MORE_FIVE','LESS_ONE','MORE_ONE']
            empl_industry_691L_unique = ['MINING', 'EDUCATION', 'RECRUITMENT', 'TRADE', 'LAWYER', 'ART_MEDIA', 'GOVERNMENT', 'TRANSPORTATION', 'MANUFACTURING', 'MARKETING', 'AGRICULTURE', 'CATERING', 'IT', 'HEALTH', 'WELNESS', 'INSURANCE', 'REAL_ESTATE', 'GAMING', 'ARMY_POLICE', 'POST_TELCO', 'FINANCE', 'OTHER', 'TOURISM', 'CHARITY_RELIGIOUS']
            empl_industry_691L_unstable = ['CHARITY_RELIGIOUS','GAMING','RECRUITMENT','WELNESS','TOURISM','AGRICULTURE','TRADE','REAL_ESTATE']
            familystate_447L_unique = ['DIVORCED','WIDOWED','MARRIED','SINGLE', 'LIVING_WITH_PARTNER']
            incometype_1044T_unique = ['SALARIED_GOVT', 'HANDICAPPED_2', 'EMPLOYED', 'PRIVATE_SECTOR_EMPLOYEE', 'SELFEMPLOYED', 'HANDICAPPED', 'RETIRED_PENSIONER', 'HANDICAPPED_3', 'OTHER']
            language1_981M_unique = ['P209_127_106', 'P10_39_147']
            relationshiptoclient_unique = ['SIBLING', 'NEIGHBOR', 'FRIEND', 'OTHER_RELATIVE', 'CHILD', 'OTHER', 'GRAND_PARENT','PARENT', 'SPOUSE', 'COLLEAGUE']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(

                # Select first non-null value of childnum_185L
                pl.col('childnum_185L').first().cast(pl.Float64,strict=False).cast(pl.Int16).alias('childnum_185L'),

                # Number of persons indicated in the application form
                pl.when( pl.col('personindex_1023L').is_not_null() ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_personindex_1023L"),

                # Sum of one-hot-encoded columns
                *[pl.col(f"persontype_1072L_{str(perstype)}").sum().cast(pl.Int16).alias(f"persontype_{str(perstype)}_1072L") for perstype in persontype_1072L_unique],
                pl.col('persontype_1072L_null').sum().cast(pl.Int16).alias('persontype_null_1072L'),
                *[pl.col(f"persontype_792L_{str(perstype)}").sum().cast(pl.Int16).alias(f"persontype_{str(perstype)}_792L") for perstype in persontype_1072L_unique],
                pl.col('persontype_792L_null').sum().cast(pl.Int16).alias('persontype_null_792L'),

                # Sum of one-hot-encoded columns
                #*[pl.col(f"education_927M_{edu}").sum().cast(pl.Int16).alias(f"education_{edu}_927M") for edu in education_927M_unique],

                # Sum of one-hot-encoded columns
                *[pl.col(f"empl_employedtotal_800L_{empl}").sum().cast(pl.Int16).alias(f"employedtotal_{empl}_800L") for empl in empl_employedtotal_800L_unique],

                # First of one-hot-encoded columns
                #*[pl.col(f"empl_industry_691L_{empl}").sum().cast(pl.Int16).alias(f"empl_industry_{empl}_691L") for empl in empl_industry_691L_unique],

                # 1 if empl_industry_691L value is in list empl_industry_691L_unstable, otherwise 0
                pl.col('empl_industry_691L').first().is_in(empl_industry_691L_unstable).cast(pl.Int16).alias('empl_industry_691L_unstable'),

                # First of one-hot-encoded column familystate_447L (TODO: pl.col(col).drop_nulls().first() if creating other features)
                #*[pl.col(f"familystate_447L_{familystate}").sum().cast(pl.Int16).alias(f"familystate_{familystate}_447L") for familystate in familystate_447L_unique]
                #*[( pl.col(f"familystate_447L_{familystate}") + pl.col(f"maritalst_703L_{familystate}") ).sum().cast(pl.Int16).alias(f"familystate_{familystate}_447L") for familystate in familystate_447L_unique],

                # one-hot-encoded columns for relationshiptoclient_415T and relationshiptoclient_642T
                #*[( pl.col(f"relationshiptoclient_415T_{rel}") + pl.col(f"relationshiptoclient_642T_{rel}") ).sum().cast(pl.Int16).alias(f"relationshiptoclient_{rel}_415T") for rel in relationshiptoclient_unique],


                # one-hot-encoded gender_992L and sex_738L
                #pl.col("gender_992L_M").first().fill_null(0).cast(pl.Int16).alias("gender_992L_M"),
                #pl.col("gender_992L_F").first().fill_null(0).cast(pl.Int16).alias("gender_992L_F"),
                #pl.col("gender_992L_null").first().fill_null(0).cast(pl.Int16).alias("gender_992L_null"),
                #pl.col("sex_738L_M").first().fill_null(0).cast(pl.Int16).alias("sex_738L_M"),
                #pl.col("sex_738L_F").first().fill_null(0).cast(pl.Int16).alias("sex_738L_F"),

                pl.col("sex_738L").drop_nulls().first().replace({"F": 0, "M": 1}, default=None).fill_null(-1).cast(pl.Int16).alias('sex_738L'),

                # one-hot-encoded columns for incometype_1044T
                #*[pl.col(f"incometype_1044T_{incometype}").sum().cast(pl.Int16).alias(f"incometype_{incometype}_1044T") for incometype in incometype_1044T_unique],
                
                # one-hot-encoded columns for language1_981M
                #*[pl.col(f"language1_981M_{language}").sum().cast(pl.Int16).alias(f"language1_{language}_981M") for language in language1_981M_unique],
                

                # Date of birth: select the first non-null value. TODO: both columns should be combine in one as they have the same date
                pl.col('birth_259D').max().alias('birth_259D'),
                pl.col('birthdate_87D').max().alias('birthdate_87D'),


                # # Encoded addresses (categorical)
                pl.col("contaddr_district_15M").first().str.replace(r'[^\d]', '').str.to_integer(strict=False).alias("contaddr_district_15M"),
                pl.col("contaddr_zipcode_807M").first().str.replace(r'[^\d]', '').str.to_integer(strict=False).alias("contaddr_zipcode_807M"),
                pl.col("empladdr_zipcode_114M").first().str.replace(r'[^\d]', '').str.to_integer(strict=False).alias("empladdr_zipcode_114M"),

                pl.col("contaddr_matchlist_1032L").drop_nulls().first().cast(pl.Int16,strict=False).alias("contaddr_matchlist_1032L"),
                pl.col("contaddr_smempladdr_334L").drop_nulls().first().cast(pl.Int16,strict=False).alias("contaddr_smempladdr_334L"),

                # is employed?
                pl.when( pl.col('empl_employedfrom_271D').drop_nulls().first().is_not_null() ).then(1).otherwise(0).cast(pl.Int16).alias("empl_employedfrom_271D_isemployed"),
                # Date of employment
                pl.col("empl_employedfrom_271D").drop_nulls().first().alias("empl_employedfrom_271D"),

                # Main income amount (TODO: more features are possible)
                pl.col("mainoccupationinc_384A").drop_nulls().first().alias("mainoccupationinc_384A"),

                # Categorical with many categories (encoded by hand to integer) - registaddr_zipcode_184M ignored
                pl.col("registaddr_district_1083M").first().alias("registaddr_district_1083M"),

                # Bool type remitter_829L
                pl.col("remitter_829L").drop_nulls().first().cast(pl.Int16,strict=False).fill_null(-1).alias("remitter_829L"),

                # Bool type safeguarantyflag_411L
                pl.col("safeguarantyflag_411L").drop_nulls().first().cast(pl.Int16,strict=False).fill_null(-1).alias("safeguarantyflag_411L"),

                # Number of type_25L indicated
                pl.when( pl.col('type_25L').is_not_null() ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_type_25L"),


                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                #*[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_last")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_last")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_mode")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_mode")],


                # Columns from person_2 table

                # Number of non-null related person roles indicated
                pl.col('relatedpersons_role_762T_encoded_max').sum().alias('num_relatedpersons_role_762T_encoded_max'),
                # The most influential role (relatedpersons_role_762T_encoded)
                pl.col('relatedpersons_role_762T_encoded_max').max().alias('relatedpersons_role_762T_encoded_max'),
                # Start date of employment
                pl.col("empls_employedfrom_796D").drop_nulls().first().alias('empls_employedfrom_796D')

            ).with_columns(
                pl.max_horizontal(pl.col('empls_employedfrom_796D'), pl.col('empl_employedfrom_271D')).alias('empls_employedfrom_796D_271D'),
                pl.max_horizontal(pl.col('birth_259D'), pl.col('birthdate_87D')).alias('birth_259D_87D'),

            ).drop(['empls_employedfrom_796D','empl_employedfrom_271D','birth_259D','birthdate_87D'])
            # Ignored: isreference_387L,registaddr_zipcode_184M,role_993L

        if table_name=='applprev_1':

            credtype_587L_unique = ['REL','CAL','COL']

            # Columns to comute Summary Statistics (max, sum, mean, median)
            summary_columns = ['pmtnum_8L']
            mean_columns = ['credamount_590A','currdebt_94A','downpmt_134A','revolvingaccount_394A','tenor_203L','maxdpdtolerance_577P',
                            'actualdpd_943P','annuity_853A','credacc_transactions_402L']
            sum_columns = ['actualdpd_943P','annuity_853A','credacc_transactions_402L','credamount_590A','currdebt_94A']
            max_columns = ['actualdpd_943P','annuity_853A','credacc_maxhisbal_375A','credacc_minhisbal_90A','currdebt_94A','downpmt_134A','mainoccupationinc_437A',
                           'maxdpdtolerance_577P']
            min_columns = []
            std_columns = []
            number_non0s_column = ['actualdpd_943P','annuity_853A','downpmt_134A']

            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(

                # Number of non-null entries in summary columns
                *[pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_"+col) for col in summary_columns],

                # Number of non-null entries and non-zeros in number_non0s_column columns
                *[pl.when(
                    (pl.col(col).is_not_null()) & (pl.col(col).cast(pl.Float64).gt(0.0))
                    ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_"+col) for col in number_non0s_column],

                # Create new features from summary columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in summary_columns],
                
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in summary_columns],

                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in summary_columns],

                # Create mean values for columns in mean_columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().alias(col+"_mean") for col in mean_columns],

                # Create std values for columns in std_columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).std().alias(col+"_std") for col in std_columns],

                # Create columns with sum values from sum_columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().alias(col+"_sum") for col in sum_columns],

                # Create columns with max values from max_columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().alias(col+"_max") for col in max_columns],

                # Create columns with min values from min_columns
                *[pl.col(col).cast(pl.Float64, strict=False).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).min().alias(col+"_min") for col in min_columns],

                # Late Payment Frequency: Count the number of instances where DPD exceeded a certain threshold (e.g., 30 days). A higher count suggests a riskier borrower.
                pl.when( (pl.col('actualdpd_943P').is_not_null()) & (pl.col('actualdpd_943P').gt(30.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_late_actualdpd_943P"),

                # Payment-to-Income Ratio: Calculate the ratio of the monthly annuity (annuity_853A) to the applicant’s income (byoccupationinc_3656910L). A higher ratio may indicate financial strain, affecting the ability to repay.
                (pl.col("annuity_853A") / pl.col("byoccupationinc_3656910L").cast(pl.Float64, strict=False).replace(1.0, None)).replace(float("inf"),None).mean().fill_nan(None).alias("annuity_853A_byoccupationinc_3656910L_ratio_mean"),
                (pl.col("annuity_853A") / pl.col("byoccupationinc_3656910L").cast(pl.Float64, strict=False).replace(1.0, None)).replace(float("inf"),None).fill_nan(None).max().alias("annuity_853A_byoccupationinc_3656910L_ratio_max"),

                # Payment-to-Income Ratio: Calculate the ratio of the monthly annuity (annuity_853A) to the applicant’s income (mainoccupationinc_437A). A higher ratio may indicate financial strain, affecting the ability to repay.
                (pl.col("annuity_853A") / pl.col("mainoccupationinc_437A")).replace(float("inf"),None).mean().fill_nan(None).alias("annuity_853A_mainoccupationinc_437A_ratio_mean"),
                (pl.col("annuity_853A") / pl.col("mainoccupationinc_437A")).replace(float("inf"),None).fill_nan(None).max().alias("annuity_853A_mainoccupationinc_437A_ratio_max"),

                pl.col("annuity_853A").drop_nulls().last().alias('annuity_853A_last'),
                pl.col("pmtnum_8L").drop_nulls().last().alias('pmtnum_8L_last'),
                

                # childnum_21L
                pl.col("childnum_21L").cast(pl.Int16, strict=False).max().alias("childnum_21L_max"),

                # Credit Utilization Ratio: Calculate the ratio of the actual balance (credacc_actualbalance_314A) to the credit limit (credacc_credlmt_575A). A high ratio may indicate credit stress.
                (pl.col("credacc_actualbalance_314A") / pl.col("credacc_credlmt_575A")).replace(float("inf"),None).fill_nan(None).mean().alias("credacc_actualbalance_314A_credacc_credlmt_575A_ratio_mean"),
                (pl.col("credacc_actualbalance_314A") / pl.col("credacc_credlmt_575A")).replace(float("inf"),None).fill_nan(None).max().alias("credacc_actualbalance_314A_credacc_credlmt_575A_ratio_max"),

                # Balance-to-Income Ratio: Divide the actual balance (credacc_actualbalance_314A) by the applicant’s income (byoccupationinc_3656910L). A high ratio may signal financial strain.
                (pl.col("credacc_actualbalance_314A") / pl.col("byoccupationinc_3656910L").cast(pl.Float64, strict=False).replace(1.0, None)).replace(float("inf"),None).mean().fill_nan(None).alias("credacc_actualbalance_314A_byoccupationinc_3656910L_ratio_mean"),
                (pl.col("credacc_actualbalance_314A") / pl.col("byoccupationinc_3656910L").cast(pl.Float64, strict=False).replace(1.0, None)).replace(float("inf"),None).fill_nan(None).max().alias("credacc_actualbalance_314A_byoccupationinc_3656910L_ratio_max"),
                # Balance-to-Income Ratio: Divide the actual balance (credacc_actualbalance_314A) by the applicant’s income (mainoccupationinc_437A). A high ratio may signal financial strain.
                (pl.col("credacc_actualbalance_314A") / pl.col("mainoccupationinc_437A")).replace(float("inf"),None).mean().fill_nan(None).alias("credacc_actualbalance_314A_mainoccupationinc_437A_ratio_mean"),

                # Check if 'mainoccupationinc_437A' increases or decreases by difference 'last - first'
                pl.when( (pl.col('mainoccupationinc_437A').last() - pl.col('mainoccupationinc_437A').first())>0 ).then(1).otherwise(0).cast(pl.Int16).alias('mainoccupationinc_437A_increase'),

                # Utilization Ratio: Divide the actual balance (credacc_actualbalance_314A) by the credit amount or card limit (credamount_590A). A high ratio suggests credit stress.
                (pl.col("credacc_actualbalance_314A") / pl.col("credamount_590A")).replace(float("inf"),None).fill_nan(None).mean().alias("credacc_actualbalance_314A_credamount_590A_ratio_mean"),
                (pl.col("credacc_actualbalance_314A") / pl.col("credamount_590A")).replace(float("inf"),None).fill_nan(None).max().alias("credacc_actualbalance_314A_credamount_590A_ratio_max"),

                # Credit Utilization Ratio: Divide the actual credit usage (e.g., outstanding debt) by the credit limit. 
                (pl.col("outstandingdebt_522A") / pl.col("credacc_credlmt_575A")).replace(float("inf"),None).fill_nan(None).mean().alias("outstandingdebt_522A_credacc_credlmt_575A_ratio_mean"),
                (pl.col("outstandingdebt_522A") / pl.col("credacc_credlmt_575A")).replace(float("inf"),None).fill_nan(None).max().alias("outstandingdebt_522A_credacc_credlmt_575A_ratio_max"),


                # 'approvaldate_319D' has pl.Date type. Compute difference between max and min dates in days
                (pl.col('approvaldate_319D').max() - pl.col('approvaldate_319D').min()).dt.total_days().mul(1.0/30).alias("approvaldate_319D_duration"),
                # Last approval date
                pl.col('approvaldate_319D').drop_nulls().last().alias('approvaldate_319D_last'),
                # Last creation date
                pl.col('creationdate_885D').drop_nulls().max().alias('creationdate_885D_last'),
                # Lat activation date
                pl.col('dateactivated_425D').drop_nulls().max().alias('dateactivated_425D_last'),
                # Last payment date
                pl.col('dtlastpmt_581D').max().alias('dtlastpmt_581D_last'),
                # Last date
                pl.col('dtlastpmtallstes_3545839D').max().alias('dtlastpmtallstes_3545839D_last'),
                # First date
                pl.col('firstnonzeroinstldate_307D').max().alias('firstnonzeroinstldate_307D_last'),
                # Employed from (TODO: add pl.col('employedfrom_700D').mean().alias('employedfrom_700D_mean'),)
                pl.col('employedfrom_700D').max().alias('employedfrom_700D_last'),
                pl.col('employedfrom_700D').min().alias('employedfrom_700D_first'),

                # Averaged duration of credits
                (pl.col('dtlastpmtallstes_3545839D') - pl.col('firstnonzeroinstldate_307D')).dt.total_days().mean().mul(1.0/30).cast(pl.Float64, strict=False).alias('firstnonzeroinstldate_307D_dtlastpmtallstes_3545839D_diff'),
                pl.when(pl.col('dtlastpmtallstes_3545839D').is_not_null()).then(1).otherwise(0).sum().cast(pl.Int64).alias("num_dtlastpmtallstes_3545839D"),
                
                # Boolean columns isbidproduct_390L and isdebitcard_527L: count true's
                pl.when( (pl.col('isbidproduct_390L').is_not_null()) & (pl.col('isbidproduct_390L').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_isbidproduct_390L"),
                pl.when( (pl.col('isdebitcard_527L').is_not_null()) & (pl.col('isdebitcard_527L').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_isdebitcard_527L"),

                # last balance on previous credit
                pl.col("credacc_actualbalance_314A").drop_nulls().last().alias("credacc_actualbalance_314A_last"),

                # Sum of one-hot-encoded columns credtype_587L
                *[pl.col(f"credtype_587L_{credtype}").sum().cast(pl.Int16).alias(f"credtype_{credtype}_587L") for credtype in credtype_587L_unique],

                ##### Depth 2 columns
                #pl.col('num_conts_type_509L_encoded').mean().alias('num_conts_type_509L_encoded_mean'),
                pl.col("conts_type_509L_encoded_max").max().alias("conts_type_509L_encoded_max"),
                #pl.col('num_credacc_cards_status_52L_encoded').mean().alias('num_credacc_cards_status_52L_encoded_mean'),
                pl.col("credacc_cards_status_52L_encoded_max").max().alias("credacc_cards_status_52L_encoded_max"),
                #pl.col('num_credacc_cards_status_52L_encoded').sum().alias('num_credacc_cards_status_52L_encoded_sum'),
                pl.col("cacccardblochreas_147M_encoded_max").max().alias("cacccardblochreas_147M_encoded_max"),


                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                #*[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Various ordinal encoded columns
                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_last")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_last")],

                *[pl.col(col).drop_nulls().last().alias(f"{col}_last") for col in data.columns if col.endswith("_encoded_mode")],
                *[pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode") for col in data.columns if col.endswith("_encoded_mode")],

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )
            # Ignore: district_544M, profession_152M

        if table_name=='debitcard_1':

            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Number of cards?
                pl.when( (pl.col('last180dayaveragebalance_704A').is_not_null()) & (pl.col('last180dayaveragebalance_704A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_last180dayaveragebalance_704A"),

                # Balance-to-Turnover Ratio:  Create a new feature by dividing last180dayaveragebalance_704A by last180dayturnover_1134A.         * A higher ratio might indicate better financial management and lower default risk.
                (pl.col("last180dayaveragebalance_704A") / pl.col("last180dayturnover_1134A")).replace(float("inf"),None).fill_nan(None).max().alias("last180dayaveragebalance_704A_last180dayturnover_1134A_ratio_max"),
                (pl.col("last180dayaveragebalance_704A") / pl.col("last180dayturnover_1134A")).replace(float("inf"),None).fill_nan(None).mean().alias("last180dayaveragebalance_704A_last180dayturnover_1134A_ratio_mean"),

                pl.col("last180dayaveragebalance_704A").sum().alias("last180dayaveragebalance_704A_sum"),
                pl.col("last180dayturnover_1134A").sum().alias("last180dayturnover_1134A_sum"),
                pl.col("last30dayturnover_651A").sum().alias("last30dayturnover_651A_sum"),

                # Recent Turnover Change: Calculate the percentage change in turnover between the last 30 days and the previous 30 days
                (pl.col('last30dayturnover_651A').mul(6.0) / pl.col('last180dayturnover_1134A')).replace(float("inf"),None).fill_nan(None).max().alias('last30dayturnover_651A_last180dayturnover_1134A_ratio_max'),
                (pl.col('last30dayturnover_651A').mul(6.0) / pl.col('last180dayturnover_1134A')).replace(float("inf"),None).fill_nan(None).mean().alias('last30dayturnover_651A_last180dayturnover_1134A_ratio_mean'),

                #  Account Age: Calculate the duration (in days) since the debit card was opened.
                pl.col('openingdate_857D').max().alias('openingdate_857D_last'),
                pl.col('openingdate_857D').min().alias('openingdate_857D_first'),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        if table_name=='deposit_1':
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Number of non-zero deposit accounts
                pl.when( (pl.col('amount_416A').is_not_null()) & (pl.col('amount_416A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_amount_416A"),
                # Accounts with zero deposit
                pl.when( pl.col('amount_416A').eq(0.0) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_amount_416A_zero"),
                # Number of closed accounts
                pl.when( pl.col('contractenddate_991D').is_not_null() ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_contractenddate_991D"),
                # Number of still open accounts
                pl.when( (pl.col('contractenddate_991D').is_not_null()) & (pl.col('amount_416A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_open_nonzero_deposits_416A"),

                # Deposit amount
                pl.col('amount_416A').sum().alias('amount_416A_sum'),
                # Sum of deposits on still open accounts
                pl.when( pl.col('contractenddate_991D').is_not_null() ).then(pl.col('amount_416A')).otherwise(0.0).sum().cast(pl.Float64).alias("amount_416A_stillopen_sum"),

                # Account Tenure: Open date of deposit account (first and last). The difference between the current date and the account opening date. Longer tenure could be associated with lower risk.
                pl.when( pl.col('contractenddate_991D').is_not_null() ).then(pl.col('openingdate_313D')).otherwise(None).min().alias("openingdate_313D_first"),
                pl.when( pl.col('contractenddate_991D').is_not_null() ).then(pl.col('openingdate_313D')).otherwise(None).max().alias("openingdate_313D_last"),

                # Averaged duration of deposit accounts (that were closed) in years
                (pl.col('contractenddate_991D') - pl.col('openingdate_313D')).dt.total_days().mean().mul(1.0/365).alias('contractenddate_991D_openingdate_313D_duration'),
                
                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        if table_name=='tax_registry_a_1':
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']
            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Tax record date (TODO: new date reference?)
                pl.col('recorddate_4527225D').max().alias('recorddate_4527225D'),

                # Number of Tax amount records
                pl.when( (pl.col('amount_4527230A').is_not_null()) & (pl.col('amount_4527230A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_amount_4527230A"),
                pl.col('amount_4527230A').sum().alias('amount_4527230A_sum'),
                pl.col('amount_4527230A').mean().alias('amount_4527230A_mean'),
                # tax grows?
                (pl.col("amount_4527230A").first() / pl.col("amount_4527230A").last()).replace(float("inf"),None).fill_nan(None).alias("amount_4527230A_first_last_ratio"),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )
            # Ignore: name_4527232M

        if table_name=='tax_registry_b_1':
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']
            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Tax record date (TODO: new date reference?)
                pl.col('deductiondate_4917603D').max().alias('deductiondate_4917603D_last'),

                # Number of Tax amount records
                pl.when( (pl.col('amount_4917619A').is_not_null()) & (pl.col('amount_4917619A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_amount_4917619A"),
                pl.col('amount_4917619A').sum().alias('amount_4917619A_sum'),
                pl.col('amount_4917619A').mean().alias('amount_4917619A_mean'),

                # Duration in days (TODO: drop?)
                ( pl.col('deductiondate_4917603D').max() - pl.col('deductiondate_4917603D').min()).dt.total_days().alias('deductiondate_4917603D_duration'),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

            )
            # Ignore: name_4917606M

        if table_name=='tax_registry_c_1':
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']
            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Tax record date (TODO: new date reference?)
                pl.col('processingdate_168D').max().alias('processingdate_168D_last'),

                # Number of Tax amount records
                pl.when( (pl.col('pmtamount_36A').is_not_null()) & (pl.col('pmtamount_36A').gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int16).alias("num_pmtamount_36A"),
                pl.col('pmtamount_36A').sum().alias('pmtamount_36A_sum'),
                pl.col('pmtamount_36A').mean().alias('pmtamount_36A_mean'),
                pl.col('pmtamount_36A').max().alias('pmtamount_36A_max'),

                # Duration in days (TODO: drop?)
                ( pl.col('processingdate_168D').max() - pl.col('processingdate_168D').min()).dt.total_days().alias('processingdate_168D_duration'),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )
            # Ignore; employername_160M

        if table_name=='other_1':
            suffixes = ['_mean_target','_frequency','_encoded','_group1', '_id', '_encoded_last','_encoded_mode']
            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                pl.col('amtdebitincoming_4809443A').sum().alias('amtdebitincoming_4809443A'),
                pl.col('amtdebitoutgoing_4809440A').sum().alias('amtdebitoutgoing_4809440A'),

                pl.col('amtdepositbalance_4809441A').sum().alias('amtdepositbalance_4809441A'),
                pl.col('amtdepositincoming_4809444A').sum().alias('amtdepositincoming_4809444A'),
                pl.col('amtdepositoutgoing_4809442A').sum().alias('amtdepositoutgoing_4809442A'),

                # Transaction Behavior Ratios: Create ratios between incoming and outgoing transactions (e.g., incoming / outgoing). Higher ratios could indicate better financial management.
                (pl.col("amtdebitincoming_4809443A").sum() / pl.col("amtdebitoutgoing_4809440A").sum()).replace(float("inf"),None).fill_nan(None).alias("amtdebitincoming_4809443A_amtdebitoutgoing_4809440A_ratio"),
                (pl.col("amtdepositincoming_4809444A").sum() / pl.col("amtdepositoutgoing_4809442A").sum()).replace(float("inf"),None).fill_nan(None).alias("amtdepositincoming_4809444A_amtdepositoutgoing_4809442A_ratio"),

                # Multiply or divide relevant columns (e.g., balance × incoming deposits).
                (pl.col("amtdepositbalance_4809441A").sum() / pl.col("amtdebitincoming_4809443A").sum()).replace(float("inf"),None).fill_nan(None).alias("amtdepositbalance_4809441A_amtdebitincoming_4809443A_ratio"),

                # Last 3 rows
                *[pl.col(col).drop_nulls().tail(3).mean().alias(f"{col}_tail_mean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                #*[pl.col(col).drop_nulls().tail(3).max().alias(f"{col}_tail_max") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],

                # Diffs
                *[pl.col(col).drop_nulls().diff().mean().alias(f"{col}_diffmean") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").std().alias(f"{col}_diffstd") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
                *[pl.col(col).drop_nulls().diff(null_behavior="drop").last().alias(f"{col}_difflast") for col in data.select(cs.numeric()).columns if not any(col.endswith(suffix) for suffix in suffixes)],
            )

        return data
    
    def add_date_features(self, data: pl.DataFrame, ref_date: str, date_cols: list) -> pl.DataFrame:
        """
        Add date features to the DataFrame:
        - data: DataFrame to add date features to
        - return: DataFrame with date features added
        """
        data = data.with_columns(                
                *[ (pl.col(ref_date) - pl.col(col)).dt.total_days().alias(col) for col in date_cols ],
        ).with_columns(
            # Convert 'date_decision' to day of week
            pl.col(ref_date).dt.weekday().cast(pl.Int8, strict=False).alias('date_decision_weekday'),
            # Convert 'date_decision' to month
            pl.col(ref_date).dt.month().cast(pl.Int8, strict=False).alias('date_decision_month'),
            # Convert 'date_decision' to quarter
            pl.col(ref_date).dt.quarter().cast(pl.Int8, strict=False).alias('date_decision_quarter'),
            # Convert 'date_decision' to week of year
            pl.col(ref_date).dt.week().cast(pl.Int16, strict=False).alias('date_decision_week'),
        ).drop(ref_date)

        return data
                

    def add_target(self, data: pl.DataFrame, train_basetable: pl.DataFrame) -> pl.DataFrame:
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

    def load_data_and_process(self) -> pl.DataFrame:

        # List of data tables to be loaded
        data_list = ['base', 'static_0', 'static_cb_0',
                     'applprev_1', 'other_1', 'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1', 'credit_bureau_a_1',
                     'credit_bureau_b_1', 'deposit_1', 'person_1', 'debitcard_1',
                     'applprev_2', 'person_2', 'credit_bureau_a_2', 'credit_bureau_b_2']


        howtojoin = 'left' if self.data_type=='test' else 'inner'

        # Read the parquet files, concat, process, aggregate and join them in chains
        #############################################################
        # Step 1: credit_bureau_b_2 -> credit_bureau_b_1 -> base
        query_credit_bureau_b_2 = (
            pl.read_parquet(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_b_2*.parquet')
            .lazy()
            .pipe(self.set_table_dtypes)
            .pipe(self.encode_categorical_columns, 'credit_bureau_b_2')
            .collect()
            .pipe(self.aggregate_depth_2, 'credit_bureau_b_2')
            .lazy()
        )

        query_credit_bureau_b_1 = (
            pl.read_parquet(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_b_1*.parquet')
            .lazy()
            .pipe(self.set_table_dtypes)
            .pipe(self.encode_categorical_columns, 'credit_bureau_b_1')
            .join(query_credit_bureau_b_2, on=["case_id", "num_group1"], how='left')  # outer
            .collect()
            .pipe(self.aggregate_depth_1, 'credit_bureau_b_1')
            .pipe(self.reduce_memory_usage_pl)
            .lazy()
        )

        query_base = (
            pl.read_parquet(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_base.parquet')
            .lazy()
            .join(query_credit_bureau_b_1, on="case_id", how=howtojoin) ## left, inner
            .collect()
        )

        #############################################################
        # Step 2: credit_bureau_b_2 -> credit_bureau_b_1 -> base
        dataframes_credit_bureau_a_2 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_a_2*.parquet')):
            #if ifile>1: continue
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'credit_bureau_a_2')
                .pipe(self.aggregate_depth_2, 'credit_bureau_a_2')
                .lazy()
            )
            dataframes_credit_bureau_a_2.append(q.collect())

        # Concat the dataframes
        query_credit_bureau_a_2 = pl.concat(dataframes_credit_bureau_a_2, how='vertical_relaxed').lazy()
        del dataframes_credit_bureau_a_2

        dataframes_credit_bureau_a_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_a_1*.parquet')):
            #if ifile>1: continue
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'credit_bureau_a_1')
                .join(query_credit_bureau_a_2, on=["case_id", "num_group1"], how='left')
                .collect()
                .pipe(self.aggregate_depth_1, 'credit_bureau_a_1')
                .lazy()
            )
            dataframes_credit_bureau_a_1.append(q.collect())

        # Concat the dataframes
        query_credit_bureau_a_1 = pl.concat(dataframes_credit_bureau_a_1, how='vertical_relaxed').pipe(self.reduce_memory_usage_pl)
        del dataframes_credit_bureau_a_1

        # Join with base
        query_base = query_base.join(query_credit_bureau_a_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 3: person_2 -> person_1 -> base
        dataframes_person_2 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_person_2*.parquet')):
            #if ifile>1: continue
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'person_2')
                .pipe(self.aggregate_depth_2, 'person_2')
                .lazy()
            )
            dataframes_person_2.append(q.collect())

        # Concat the dataframes
        query_person_2 = pl.concat(dataframes_person_2, how='vertical_relaxed').lazy()
        del dataframes_person_2

        dataframes_person_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_person_1*.parquet')):
            #if ifile>1: continue
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'person_1')
                .join(query_person_2, on=["case_id", "num_group1"], how='left')
                .collect()
                .pipe(self.aggregate_depth_1, 'person_1')
                .lazy()
            )
            dataframes_person_1.append(q.collect())

        # Concat the dataframes
        query_person_1 = pl.concat(dataframes_person_1, how='vertical_relaxed').pipe(self.reduce_memory_usage_pl)
        del dataframes_person_1

        # Join with base
        query_base = query_base.join(query_person_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 4: applprev_2 -> applprev_1 => base
        dataframes_applprev_2 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_applprev_2*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'applprev_2')
                .pipe(self.aggregate_depth_2, 'applprev_2')
                .lazy()
            )
            dataframes_applprev_2.append(q.collect())

        # Concat the dataframes
        query_applprev_2 = pl.concat(dataframes_applprev_2, how='vertical_relaxed').lazy()
        del dataframes_applprev_2

        dataframes_applprev_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_applprev_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'applprev_1')
                .join(query_applprev_2, on=["case_id", "num_group1"], how='left')
                .collect()
                .pipe(self.aggregate_depth_1, 'applprev_1')
                .lazy()
            )
            dataframes_applprev_1.append(q.collect())

        # Concat the dataframes
        query_applprev_1 = pl.concat(dataframes_applprev_1, how='vertical_relaxed').pipe(self.reduce_memory_usage_pl)
        del dataframes_applprev_1

        # Join with base
        query_base = query_base.join(query_applprev_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 5:  debitcard_1 -> base
        dataframes_debitcard_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_debitcard_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .collect()
                .pipe(self.aggregate_depth_1, 'debitcard_1')
                .lazy()
            )
            dataframes_debitcard_1.append(q.collect())

        # Concat the dataframes
        query_debitcard_1 = pl.concat(dataframes_debitcard_1, how='vertical_relaxed')
        del dataframes_debitcard_1

        # Join with base
        query_base = query_base.join(query_debitcard_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 6: deposit_1 -> base
        dataframes_deposit_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_deposit_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                #.pipe(self.encode_categorical_columns, 'deposit_1')
                .collect()
                .pipe(self.aggregate_depth_1, 'deposit_1')
                .lazy()
            )
            dataframes_deposit_1.append(q.collect())

        # Concat the dataframes
        query_deposit_1 = pl.concat(dataframes_deposit_1, how='vertical_relaxed')
        del dataframes_deposit_1

        # Join with base
        query_base = query_base.join(query_deposit_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 7: tax_registry_a_1 -> base
        dataframes_tax_registry_a_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_tax_registry_a_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .collect()
                .pipe(self.aggregate_depth_1, 'tax_registry_a_1')
                .lazy()
            )
            dataframes_tax_registry_a_1.append(q.collect())

        # Concat the dataframes
        query_tax_registry_a_1 = pl.concat(dataframes_tax_registry_a_1, how='vertical_relaxed')
        del dataframes_tax_registry_a_1

        # Join with base
        query_base = query_base.join(query_tax_registry_a_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 8: tax_registry_b_1 -> base
        dataframes_tax_registry_b_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_tax_registry_b_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .collect()
                .pipe(self.aggregate_depth_1, 'tax_registry_b_1')
                .lazy()
            )
            dataframes_tax_registry_b_1.append(q.collect())

        # Concat the dataframes
        query_tax_registry_b_1 = pl.concat(dataframes_tax_registry_b_1, how='vertical_relaxed')
        del dataframes_tax_registry_b_1

        # Join with base
        query_base = query_base.join(query_tax_registry_b_1, on="case_id", how=howtojoin)

        #############################################################
        # Step 9: tax_registry_c_1 -> base
        dataframes_tax_registry_c_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_tax_registry_c_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                #.pipe(self.encode_categorical_columns, 'tax_registry_c_1')
                .collect()
                .pipe(self.aggregate_depth_1, 'tax_registry_c_1')
                .lazy()
            )
            dataframes_tax_registry_c_1.append(q.collect())

        # Concat the dataframes
        query_tax_registry_c_1 = pl.concat(dataframes_tax_registry_c_1, how='vertical_relaxed')
        del dataframes_tax_registry_c_1

        # Join with base
        query_base = query_base.join(query_tax_registry_c_1, on="case_id", how=howtojoin)
        #############################################################
        # Step 10 other_1 -> base
        dataframes_other_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_other_1*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .collect()
                .pipe(self.aggregate_depth_1, 'other_1')
                .lazy()
            )
            dataframes_other_1.append(q.collect())

        # Concat the dataframes
        query_other_1 = pl.concat(dataframes_other_1, how='vertical_relaxed')
        del dataframes_other_1

        # Join with base
        query_base = query_base.join(query_other_1, on="case_id", how=howtojoin)
        #############################################################
        # Step 11: static_0 -> base
        dataframes_static_0 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_static_0*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'static_0')
                .collect()
                .lazy()
            )
            dataframes_static_0.append(q.collect())

        # Concat the dataframes
        query_static_0 = pl.concat(dataframes_static_0, how='vertical_relaxed').pipe(self.reduce_memory_usage_pl)
        del dataframes_static_0

        # Join with base
        query_base = query_base.join(query_static_0, on="case_id", how=howtojoin)
        #############################################################
        # Step 12: static_cb_0 -> base
        dataframes_static_cb_0 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_static_cb_0*.parquet')):
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'static_cb_0')
                .collect()
                .lazy()
            )
            dataframes_static_cb_0.append(q.collect())

        # Concat the dataframes
        query_static_cb_0 = pl.concat(dataframes_static_cb_0, how='vertical_relaxed').pipe(self.reduce_memory_usage_pl)
        del dataframes_static_cb_0

        # Join with base
        query_base = query_base.join(query_static_cb_0, on="case_id", how=howtojoin)
        #############################################################


        # Process the pl.Date columns
        # date features to be transformed from pl.Date using reference date
        date_cols = ['empls_employedfrom_796D_271D','birth_259D_87D','refreshdate_3813885D_max',
                     'approvaldate_319D_last','creationdate_885D_last',
                    'dateactivated_425D_last','dtlastpmt_581D_last','dtlastpmtallstes_3545839D_last','firstnonzeroinstldate_307D_last',
                    'employedfrom_700D_last','employedfrom_700D_first',
                    'openingdate_857D_last', 'openingdate_857D_first', 'openingdate_313D_first', 'openingdate_313D_last','recorddate_4527225D',
                    'deductiondate_4917603D_last','processingdate_168D_last',
                    'dateofcredstart_739D_mean','dateofcredstart_181D_mean','dateofrealrepmt_138D_mean','dateofcredend_353D_mean',
                    'overdueamountmax2date_1142D_max','numberofoverdueinstlmaxdat_148D_max','numberofoverdueinstlmaxdat_641D_mean',
                    'overdueamountmax2date_1002D_mean','dateofcredstart_181D_max','dateofcredend_289D_mean','dateofcredend_353D_max']
        date_cols += predata.date_static_0_columns
        date_cols += predata.date_static_cb_0_columns
        date_ref = 'date_decision'    # refreshdate_3813885D_max

        # Convert 'date_decision' to pl.Date
        query_base = query_base.with_columns(pl.col('date_decision').str.strptime(pl.Date, "%Y-%m-%d").alias('date_decision'))
        query_base = query_base.pipe(self.add_date_features, date_ref, date_cols)

        # Remove all-null-rows in train data
        if self.data_type=='train':
           cols_pred = [col for col in query_base.columns if col not in ['target','case_id','WEEK_NUM','MONTH','date_decision']]
           query_base = query_base.filter(~pl.all_horizontal(pl.col(*cols_pred).is_null()))

        # Fill all null values and NaN values with 0
        query_base = query_base.fill_nan(None)   # .fill_null(0)   .with_columns(cs.numeric().replace(float("inf"),0.0))

        # Drop these categorical features. Drop columns that were in the tail of LGBM's feature importance
        drop_cat_cols = ['previouscontdistrict_112M_encoded', 'contaddr_zipcode_807M_encoded_mode', 'contaddr_zipcode_807M_encoded_last',
                         'contaddr_district_15M_encoded_last', 'contaddr_district_15M_encoded_mode', 'contaddr_district_15M_mean_target',
                         'previouscontdistrict_112M_mean_target', 'contaddr_zipcode_807M_mean_target',
                         'empladdr_district_926M_encoded_last','empladdr_district_926M_encoded_mode','empladdr_zipcode_114M_encoded_mode','empladdr_zipcode_114M_encoded_last']
        singleval_cols = ['amount_1115A_credlmt_3940954A_ratio_max', 'amount_1115A_credlmt_1052A_ratio_max', 'residualamount_488A_std', 'persontype_1.0_792L', 'deferredmnthsnum_166L']
        query_base = query_base.drop(drop_cat_cols+singleval_cols) #.drop(predata.drop_cols_by_importance)

        for col in query_base.columns:
            if col in predata.drop_cols_by_importance:
                query_base = query_base.drop(col)

        # Drop the columns in query_base that end with "_mean_target"
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_mean_target')])
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_encoded_mode')])
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_encoded_mode_last')])
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_encoded_mode_mode')])
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_encoded_last_mode')])
        query_base = query_base.drop([col for col in query_base.columns if col.endswith('_encoded_max_mode')])

        # Final memory optimization
        print('Final memory optimization')
        query_base = query_base.pipe(self.reduce_memory_usage_pl)


        return query_base # query_base


# Main function here
if __name__ == "__main__":

    dataPath = "/kaggle/input/home-credit-credit-risk-model-stability/"

    # Create a CreditRisk object
    cr = CreditRiskProcessing(data_path=dataPath, data_type="train", complete=True)

    # Load the data
    data_store = cr.load_data_and_process()

    


    # Create the features
    features = cr.create_features(data_store)

    # Add the target column
    features = cr.add_target(features, data_store['base'])

    # Save the features to a parquet file
    features.write_parquet('features.parquet')