import os, glob

import polars as pl
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
            #roles_encoding = predata.relatedpersons_role_762T_encoding
            #data = self.cat_to_int_encode(data, "relatedpersons_role_762T", "relatedpersons_role_encoded", roles_encoding)

            # Chain an addition of several new columns
            # TODO: mormalize frequency by norm_frequency(predata.relatedpersons_role_762T_mean_target)
            # TODO: add another fill_null('None') in the end of chain?
            data = data.with_columns(
                # Categorical ordered encoding
                pl.col("relatedpersons_role_762T").replace(predata.relatedpersons_role_762T_encoding, default=None).alias("relatedpersons_role_encoded"),

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
            # contact_type_encoding = predata.conts_type_509L_encoding
            # data = self.cat_to_int_encode(data, "conts_type_509L", "conts_type_encoded", contact_type_encoding)
            # credacc_cards_status_encoding = predata.credacc_cards_status_52L_encoding
            # data = self.cat_to_int_encode(data, "credacc_cards_status_52L", "credacc_cards_status_encoded", credacc_cards_status_encoding)
            # cacccardblochreas_147M_encoding = predata.cacccardblochreas_147M_encoding
            # data = self.cat_to_int_encode(data, "cacccardblochreas_147M", "cacccardblochreas_encoded", cacccardblochreas_147M_encoding)

            # Adding new columns
            data = data.with_columns(
                # Categorical ordered encoding
                pl.col("conts_type_509L").replace(predata.conts_type_509L_encoding, default=None).alias("conts_type_encoded"),
                pl.col("credacc_cards_status_52L").replace(predata.credacc_cards_status_52L_encoding, default=None).alias("credacc_cards_status_encoded"),
                pl.col("cacccardblochreas_147M").replace(predata.cacccardblochreas_147M_encoding, default=None).alias("cacccardblochreas_encoded"),

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
                pl.col("pmts_pmtsoverdue_635A").fill_null(0).alias("pmts_pmtsoverdue_635A"),
            )

        if table_name=='credit_bureau_b_1':
            data = data.with_columns(
                pl.col("periodicityofpmts_997L").fill_null('None').replace(predata.periodicityofpmts_997L_mean_target, default=None).alias("periodicityofpmts_997L_mean_target"),
                pl.col("periodicityofpmts_997L").fill_null('None').replace(predata.periodicityofpmts_997L_frequency, default=None).alias("periodicityofpmts_997L_frequency"),
                pl.col("periodicityofpmts_997L").fill_null('None').replace(predata.periodicityofpmts_997L_interval, default=None).alias("periodicityofpmts_997L_interval"),

                pl.col("classificationofcontr_1114M").replace(predata.classificationofcontr_1114M_mean_target, default=None).alias("classificationofcontr_1114M_mean_target"),
                pl.col("classificationofcontr_1114M").replace(predata.classificationofcontr_1114M_frequency, default=None).alias("classificationofcontr_1114M_frequency"),

                pl.col("contractst_516M").replace(predata.contractst_516M_mean_target, default=None).alias("contractst_516M_mean_target"),
                pl.col("contractst_516M").replace(predata.contractst_516M_frequency, default=None).alias("contractst_516M_frequency"),

                pl.col("contracttype_653M").replace(predata.contracttype_653M_mean_target, default=None).alias("contracttype_653M_mean_target"),
                pl.col("contracttype_653M").replace(predata.contracttype_653M_frequency, default=None).alias("contracttype_653M_frequency"),

                pl.col("credor_3940957M").replace(predata.credor_3940957M_mean_target, default=None).alias("credor_3940957M_mean_target"),
                pl.col("credor_3940957M").replace(predata.credor_3940957M_frequency, default=None).alias("credor_3940957M_frequency"),

                pl.col("periodicityofpmts_997M").fill_null('None').replace(predata.periodicityofpmts_997M_mean_target, default=None).alias("periodicityofpmts_997M_mean_target"),
                pl.col("periodicityofpmts_997M").fill_null('None').replace(predata.periodicityofpmts_997M_frequency, default=None).alias("periodicityofpmts_997M_frequency"),

                
                pl.col("pmtmethod_731M").replace(predata.pmtmethod_731M_mean_target, default=None).alias("pmtmethod_731M_mean_target"),
                pl.col("pmtmethod_731M").replace(predata.pmtmethod_731M_frequency, default=None).alias("pmtmethod_731M_frequency"),

                pl.col("purposeofcred_722M").replace(predata.purposeofcred_722M_mean_target, default=None).alias("purposeofcred_722M_mean_target"),
                pl.col("purposeofcred_722M").replace(predata.purposeofcred_722M_frequency, default=None).alias("purposeofcred_722M_frequency"),

                pl.col("subjectrole_326M").replace(predata.subjectrole_326M_mean_target, default=None).alias("subjectrole_326M_mean_target"),
                pl.col("subjectrole_326M").replace(predata.subjectrole_326M_frequency, default=None).alias("subjectrole_326M_frequency"),

                pl.col("subjectrole_43M").replace(predata.subjectrole_43M_mean_target, default=None).alias("subjectrole_43M_mean_target"),
                pl.col("subjectrole_43M").replace(predata.subjectrole_43M_frequency, default=None).alias("subjectrole_43M_frequency"),
            )

        if table_name=='credit_bureau_a_1':
            data = data.with_columns(
                pl.col("classificationofcontr_13M").replace(predata.classificationofcontr_13M_mean_target, default=None).alias("classificationofcontr_13M_mean_target"),
                pl.col("classificationofcontr_13M").replace(predata.classificationofcontr_13M_frequency, default=None).alias("classificationofcontr_13M_frequency"),

                pl.col("classificationofcontr_400M").replace(predata.classificationofcontr_400M_mean_target, default=None).alias("classificationofcontr_400M_mean_target"),
                pl.col("classificationofcontr_400M").replace(predata.classificationofcontr_400M_frequency, default=None).alias("classificationofcontr_400M_frequency"),

                pl.col("contractst_545M").replace(predata.contractst_545M_mean_target, default=None).alias("contractst_545M_mean_target"),
                pl.col("contractst_545M").replace(predata.contractst_545M_frequency, default=None).alias("contractst_545M_frequency"),

                pl.col("contractst_964M").replace(predata.contractst_964M_mean_target, default=None).alias("contractst_964M_mean_target"),
                pl.col("contractst_964M").replace(predata.contractst_964M_frequency, default=None).alias("contractst_964M_frequency"),

                pl.col("description_351M").replace(predata.description_351M_mean_target, default=None).alias("description_351M_mean_target"),
                pl.col("description_351M").replace(predata.description_351M_frequency, default=None).alias("description_351M_frequency"),

                pl.col("financialinstitution_382M").replace(predata.financialinstitution_382M_mean_target, default=None).alias("financialinstitution_382M_mean_target"),
                pl.col("financialinstitution_382M").replace(predata.financialinstitution_382M_frequency, default=None).alias("financialinstitution_382M_frequency"),

                pl.col("financialinstitution_591M").replace(predata.financialinstitution_591M_mean_target, default=None).alias("financialinstitution_591M_mean_target"),
                pl.col("financialinstitution_591M").replace(predata.financialinstitution_591M_frequency, default=None).alias("financialinstitution_591M_frequency"),

                pl.col("purposeofcred_426M").replace(predata.purposeofcred_426M_mean_target, default=None).alias("purposeofcred_426M_mean_target"),
                pl.col("purposeofcred_426M").replace(predata.purposeofcred_426M_frequency, default=None).alias("purposeofcred_426M_frequency"),

                pl.col("purposeofcred_874M").replace(predata.purposeofcred_874M_mean_target, default=None).alias("purposeofcred_874M_mean_target"),
                pl.col("purposeofcred_874M").replace(predata.purposeofcred_874M_frequency, default=None).alias("purposeofcred_874M_frequency"),

                pl.col("subjectrole_182M").replace(predata.subjectrole_182M_mean_target, default=None).alias("subjectrole_182M_mean_target"),
                pl.col("subjectrole_182M").replace(predata.subjectrole_182M_frequency, default=None).alias("subjectrole_182M_frequency"),

                pl.col("subjectrole_93M").replace(predata.subjectrole_93M_mean_target, default=None).alias("subjectrole_93M_mean_target"),
                pl.col("subjectrole_93M").replace(predata.subjectrole_93M_frequency, default=None).alias("subjectrole_93M_frequency"),
            )

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
            
        elif table_name=='applprev_2':
            # Encode categorical columns
            #data = self.encode_categorical_columns(data, table_name)

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
            
        elif table_name=='credit_bureau_a_2':
            # Encode categorical columns
            #data = self.encode_categorical_columns(data, table_name)

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
            
        elif table_name=='credit_bureau_b_2':
            # Fill null for pmts_dpdvalue_108P (observed)
            #data = self.encode_categorical_columns(data, table_name)

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

        return data
    
    def aggregate_depth_1(self, data: pl.DataFrame, table_name: str) -> pl.DataFrame:
        """
        Aggregate data by case_id
        """
        if table_name=='credit_bureau_b_1':
            # Encoding categorical columns
            #data = self.encode_categorical_columns(data, table_name)

            # Columns to comute Summary Statistics (max, sum, mean, median)
            summary_columns = ['amount_1115A', 'totalamount_503A', 'totalamount_881A', 'overdueamountmax_950A',
                               'interesteffectiverate_369L','interestrateyearly_538L', 'pmtdaysoverdue_1135P', 'pmtnumpending_403L']
            mean_columns = ['installmentamount_644A', 'installmentamount_833A', 'residualamount_1093A', 'residualamount_3940956A','instlamount_892A','debtvalue_227A','debtpastduevalue_732A',
                            'dpd_550P','dpd_733P','dpdmax_851P', 'residualamount_127A','numberofinstls_810L']
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
                *[pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_"+col) for col in summary_columns],

                # Create new features from summary columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().fill_null(0.0).alias(col+"_max") for col in summary_columns],
                
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().fill_null(0.0).alias(col+"_sum") for col in summary_columns],

                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().fill_null(0.0).alias(col+"_mean") for col in summary_columns],

                #*[pl.col(col).filter(
                #        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                #        ).median().fill_null(0.0).alias(col+"_median") for col in summary_columns],

                # Create mean values for columns in mean_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().fill_null(0.0).alias(col+"_mean") for col in mean_columns],

                # Create columns with sum values from sum_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().fill_null(0.0).alias(col+"_sum") for col in sum_columns],

                # Create columns with max values from max_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().fill_null(0.0).alias(col+"_max") for col in max_columns],

                # Create columns with min values from min_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).min().fill_null(0.0).alias(col+"_min") for col in min_columns],


                # A binary feature indicating whether a column has at least one missing value (null)
                (pl.col("amount_1115A").is_null()).any().alias("amount_1115A_null"),
                (pl.col("totalamount_881A").is_null()).any().alias("totalamount_881A_null"),
                
                # Credit Utilization Ratio: ratio of amount_1115A to totalamount_503A 
                (pl.col("amount_1115A") / pl.col("totalamount_503A")).fill_null(0.0).replace(float("inf"),0.0).max().alias("amount_1115A_totalamount_503A_ratio_max"),
                (pl.col("amount_1115A").fill_null(0.0).max() / pl.col("credlmt_3940954A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("amount_1115A_credlmt_3940954A_ratio"),
                (pl.col("amount_1115A").fill_null(0.0) / pl.col("credlmt_3940954A").fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).max().alias("amount_1115A_credlmt_3940954A_ratio_max"),
                (pl.col("amount_1115A").fill_null(0.0).max() / pl.col("credlmt_1052A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("amount_1115A_credlmt_1052A_ratio"),
                (pl.col("amount_1115A").fill_null(0.0) / pl.col("credlmt_1052A").fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).max().alias("amount_1115A_credlmt_1052A_ratio_max"),
        

                # Compute the absolute difference between totalamount_503A and amount_1115A.
                (pl.col("totalamount_503A").fill_null(0.0) - pl.col("amount_1115A").fill_null(0.0)).abs().max().alias("totalamount_503A_max_diff"),
                # Difference in Credit Limits
                (pl.col("credlmt_3940954A").fill_null(0.0) - pl.col("credlmt_228A").fill_null(0.0)).max().alias("credlmt_3940954A_credlmt_228A_diff_max"),
                (pl.col("credlmt_3940954A").fill_null(0.0) - pl.col("credlmt_1052A").fill_null(0.0)).max().alias("credlmt_3940954A_credlmt_1052A_diff_max"),
                (pl.col("credlmt_3940954A").fill_null(0.0).max() - pl.col("credlmt_228A").fill_null(0.0).max()).alias("credlmt_3940954A_credlmt_228A_diff"),
                (pl.col("credlmt_3940954A").fill_null(0.0).max() - pl.col("credlmt_1052A").fill_null(0.0).max()).alias("credlmt_3940954A_credlmt_1052A_diff"),

                # TODO: Log Transformations: If the credit amounts have a wide range, consider taking the logarithm to normalize the distribution.

                # Max value of totalamount_503A over totalamount_881A
                (pl.col("totalamount_503A").fill_null(0.0).max() / pl.col("totalamount_881A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("totalamount_503A_881A"),

                ## Columns with years
                #*[pl.col(col).cast(pl.Float64).max().fill_null(0.0).alias(col+"_last") for col in year_columns],
                #*[(pl.col(col).cast(pl.Float64).max() - pl.col(col).cast(pl.Float64).min()).fill_null(0.0).alias(col+"_duration") for col in year_columns],

                # Overdue-to-Installment Ratio
                (pl.col("overdueamountmax_950A").fill_null(0.0).max() / pl.col("installmentamount_833A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("overdueamountmax_950A_installmentamount_833A_ratio"),
                (pl.col("overdueamountmax_950A").fill_null(0.0).max() / pl.col("instlamount_892A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("overdueamountmax_950A_instlamount_892A_ratio"),
                
                # Residual-to-Credit Limit Ratio
                (pl.col("residualamount_3940956A").fill_null(0.0).max() / pl.col("credlmt_3940954A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("residualamount_3940956A_credlmt_3940954A_ratio"),
                (pl.col("residualamount_3940956A").fill_null(0.0) / pl.col("credlmt_3940954A").fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).max().alias("residualamount_3940956A_credlmt_3940954A_ratio_max"),
        
                # Create a binary feature indicating whether the maximum debt occurred recently (e.g., within the last month or quarter) based on maxdebtpduevalodued_3940955A
                (pl.col("maxdebtpduevalodued_3940955A").filter(
                        (pl.col("maxdebtpduevalodued_3940955A").is_not_null()) & (pl.col("maxdebtpduevalodued_3940955A").gt(0.0))
                    ).min() < 120.0).fill_null(False).alias("maxdebtpduevalodued_3940955A_isrecent"),

                # TODO Debt-to-Income Ratio: Divide the outstanding debt amount (debtvalue_227A) by the client’s income (if available). This ratio provides insights into a client’s ability to manage debt relative to their income.

                # Past Due Ratio: Calculate the ratio of unpaid debt (debtpastduevalue_732A) to the total outstanding debt (debtvalue_227A). High past due ratios may indicate higher credit risk.
                (pl.col("debtpastduevalue_732A").fill_null(0.0).max() / pl.col("debtvalue_227A").fill_null(0.0).max()).replace(float("inf"),0.0).fill_nan(0.0).alias("debtpastduevalue_732A_debtvalue_227A_ratio"),
                (pl.col("debtpastduevalue_732A").fill_null(0.0) / pl.col("debtvalue_227A").fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).max().alias("debtpastduevalue_732A_debtvalue_227A_ratio_max"),


                # Number of non-null and greater than 0.0 values in dpd_550P, dpd_733P and dpdmax_851P columns
                pl.when( (pl.col("dpd_550P").is_not_null()) & (pl.col("dpd_550P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_dpd_550P"),
                pl.when( (pl.col("dpd_733P").is_not_null()) & (pl.col("dpd_733P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_dpd_733P"),
                pl.when( (pl.col("dpdmax_851P").is_not_null()) & (pl.col("dpdmax_851P").gt(0.0)) ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_dpdmax_851P"),
                

                # Columns for month (dpdmaxdatemonth_804T) and years (dpdmaxdateyear_742T)
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max() - 
                    pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).min()).dt.total_days().fill_null(0.0).alias("dpdmaxdate_duration"),
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max()).dt.year().fill_null(0.0).alias("dpdmaxdateyear_742T_last"),
                (pl.date(pl.col("dpdmaxdateyear_742T").cast(pl.Float64),pl.col("dpdmaxdatemonth_804T").cast(pl.Float64), 1).max()).dt.month().fill_null(0.0).alias("dpdmaxdatemonth_804T_last"),

                # Columns for month (overdueamountmaxdatemonth_494T) and years (overdueamountmaxdateyear_432T)
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max() - 
                    pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).min()).dt.total_days().fill_null(0.0).alias("overdueamountmaxdat_duration"),
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max()).dt.year().fill_null(0.0).alias("overdueamountmaxdateyear_432T_last"),
                (pl.date(pl.col("overdueamountmaxdateyear_432T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_494T").cast(pl.Float64), 1).max()).dt.month().fill_null(0.0).alias("overdueamountmaxdatemonth_494T_last"),

                # Durations: last update - contract date
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().min().fill_null(0.0).alias('lastupdate_260D_contractdate_551D_diff_min'),
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().max().fill_null(0.0).alias('lastupdate_260D_contractdate_551D_diff_max'),
                (pl.col("lastupdate_260D") - pl.col("contractdate_551D")).dt.total_days().mean().fill_null(0.0).alias('lastupdate_260D_contractdate_551D_diff_mean'),
                # Duration:  contract maturity date - last update
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().min().fill_null(0.0).alias('contractmaturitydate_151D_lastupdate_260D_diff_min'),
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().max().fill_null(0.0).alias('contractmaturitydate_151D_lastupdate_260D_diff_max'),
                (pl.col("contractmaturitydate_151D") - pl.col("lastupdate_260D")).dt.total_days().mean().fill_null(0.0).alias('contractmaturitydate_151D_lastupdate_260D_diff_mean'),

                # Various mean_target columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_mean_target")],
                # Various frequency columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_frequency")],
                # Interval columns
                *[pl.col(col).mean().alias(col) for col in data.columns if col.endswith("_interval")],


                # Columns from credit_bureau_b_2
                (pl.col("pmts_date_1107D_last").max() - pl.col("pmts_date_1107D_first").min()).alias("pmts_date_1107D_duration"),
                (pl.col("pmts_date_1107D_last").max() - pl.col("pmts_date_1107D_first").min()).mul(365).alias("pmts_date_1107D_duration_days"),
                pl.col("pmts_date_1107D_first").min().alias("pmts_date_1107D_first"),
                pl.col("pmts_date_1107D_last").max().alias("pmts_date_1107D_last"),

            )

        if table_name=='credit_bureau_a_1':

            # TODO
            # - 'annualeffectiverate_199L','annualeffectiverate_63L': consider aggregating by other relevant features, such as 
            #    1) loan type,
            #    2) borrower credit score, 
            #    3) loan purpose (purposeofcred_426M,purposeofcred_722M,purposeofcred_874M).
            #   This way, you can analyze how interest rates vary across different segments.


            # Columns to comute Summary Statistics (max, sum, mean, median)
            summary_columns = ['annualeffectiverate_199L','annualeffectiverate_63L','contractsum_5085717L','interestrate_508L']
            mean_columns = ['credlmt_230A','credlmt_935A','nominalrate_281L','nominalrate_498L','numberofinstls_229L','numberofinstls_320L','numberofoutstandinstls_520L','numberofoutstandinstls_59L'
                            'numberofoverdueinstls_725L','numberofoverdueinstls_834L','periodicityofpmts_1102L','periodicityofpmts_837L','prolongationcount_1120L','prolongationcount_599L',
                            'totalamount_6A','totalamount_996A']
            sum_columns = ['credlmt_230A','credlmt_935A','debtoutstand_525A','debtoverdue_47A','dpdmax_139P','dpdmax_757P','instlamount_852A','instlamount_768A',
                           'monthlyinstlamount_674A','monthlyinstlamount_332A','numberofcontrsvalue_258L','numberofcontrsvalue_358L','numberofinstls_229L','numberofinstls_320L',
                           'numberofoutstandinstls_520L','numberofoutstandinstls_59L','numberofoverdueinstlmax_1039L','outstandingamount_354A','outstandingamount_362A',
                           'overdueamount_31A','overdueamount_659A','overdueamountmax_155A','overdueamountmax_35A','prolongationcount_1120L','prolongationcount_599L',
                           'residualamount_856A','residualamount_488A','totalamount_6A','totalamount_996A','totaldebtoverduevalue_178A','totaldebtoverduevalue_718A',
                           'totaloutstanddebtvalue_39A','totaloutstanddebtvalue_668A']
                            # TODO in sum: 'numberofoverdueinstlmax_1039L','numberofoverdueinstlmax_1151L','numberofoverdueinstls_725L','numberofoverdueinstls_834L'
            max_columns = ['credlmt_230A','credlmt_935A','dpdmax_139P','dpdmax_757P','overdueamountmax2_14A','overdueamountmax2_398A','prolongationcount_1120L','prolongationcount_599L']
            min_columns = []
            std_columns = ['nominalrate_281L','nominalrate_498L','residualamount_856A','residualamount_488A']
            number_non0s_column = ['dpdmax_139P','dpdmax_757P','monthlyinstlamount_674A','monthlyinstlamount_332A','numberofoutstandinstls_520L','numberofoutstandinstls_59L',
                                   'numberofoverdueinstlmax_1151L','numberofoverdueinstls_725L','numberofoverdueinstls_834L','dateofcredend_353D','dateofcredend_289D']

            # Aggregating by case_id
            data = data.group_by('case_id').agg(
                # Number of non-null entries in summary columns
                *[pl.when(pl.col(col).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_"+col) for col in summary_columns],

                # Number of non-null entries and non-zeros in number_non0s_column columns
                *[pl.when(
                    (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                    ).then(1).otherwise(0).sum().cast(pl.Int8).alias("num_"+col) for col in number_non0s_column],

                # Create new features from summary columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().fill_null(0.0).alias(col+"_max") for col in summary_columns],
                
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().fill_null(0.0).alias(col+"_sum") for col in summary_columns],

                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().fill_null(0.0).alias(col+"_mean") for col in summary_columns],

                # Create mean values for columns in mean_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).mean().fill_null(0.0).alias(col+"_mean") for col in mean_columns],

                # Create std values for columns in std_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).std().fill_null(0.0).alias(col+"_std") for col in std_columns],

                # Create columns with sum values from sum_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).sum().fill_null(0.0).alias(col+"_sum") for col in sum_columns],

                # Create columns with max values from max_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).max().fill_null(0.0).alias(col+"_max") for col in max_columns],

                # Create columns with min values from min_columns
                *[pl.col(col).filter(
                        (pl.col(col).is_not_null()) & (pl.col(col).gt(0.0))
                        ).min().fill_null(0.0).alias(col+"_min") for col in min_columns],

                
                # Diffs
                (pl.col('annualeffectiverate_63L').sum().fill_null(0.0) - pl.col('annualeffectiverate_199L').sum().fill_null(0.0)).alias('annualeffectiverate_63L_199L_sum_diff'),
                (pl.col('annualeffectiverate_63L').mean().fill_null(0.0) - pl.col('annualeffectiverate_199L').mean().fill_null(0.0)).alias('annualeffectiverate_63L_199L_mean_diff'),

                (pl.col('credlmt_935A').sum().fill_null(0.0) - pl.col('credlmt_230A').sum().fill_null(0.0)).alias('credlmt_935A_1230A_sum_diff'),
                (pl.col('credlmt_935A').mean().fill_null(0.0) - pl.col('credlmt_230A').mean().fill_null(0.0)).alias('credlmt_935A_230A_mean_diff'),

                # Interest Rate Spread: Calculate the difference between the nominal interest rates for active and closed contracts. This spread could be indicative of risk.
                (pl.col('nominalrate_281L').mean().fill_null(0.0) - pl.col('nominalrate_498L').mean().fill_null(0.0)).alias('nominalrate_281L_498L_mean_diff'),

                # Contract Sum Spread: Calculate the difference between the sum of active and closed contracts. This spread could be indicative of risk.
                (pl.col('contractsum_5085717L').mean().fill_null(0.0) - pl.col('contractsum_5085717L').mean().fill_null(0.0)).alias('contractsum_5085717L_mean_diff'),

                # DPD Spread: Calculate the difference between the maximum DPD values for active and closed contracts. This spread could be indicative of risk.
                (pl.col('dpdmax_139P').mean().fill_null(0.0) - pl.col('dpdmax_757P').mean().fill_null(0.0)).alias('dpdmax_139P_757P_mean_diff'),

                # Instalment Difference
                (pl.col('instlamount_768A').sum().fill_null(0.0) - pl.col('instlamount_852A').sum().fill_null(0.0)).alias('instlamount_768A_852A_diff'),

                # Overdue Percentage: Calculate the percentage of overdue debt (debtoverdue_47A) relative to the total outstanding debt (debtoutstand_525A). High percentages may signal credit risk.
                (pl.col('debtoverdue_47A').sum().fill_null(0.0) / pl.col('debtoutstand_525A').sum().fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).alias('debtoverdue_47A_debtoutstand_525A_ratio'),
                
                # Debt Utilization: Divide the outstanding debt by the credit limit. High utilization ratios may indicate risk.
                (pl.col('debtoutstand_525A').sum().fill_null(0.0) / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('debtoutstand_525A_credlmt_935A_ratio'),

                # Instalment Coverage Ratio: Divide the total amount paid (instlamount_852A) by the total amount due (instlamount_768A). A higher ratio suggests better payment behavior.
                (pl.col('instlamount_852A').sum().fill_null(0.0) / pl.col('instlamount_768A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('instlamount_852A_768A_ratio'),

                # Instalment Difference: Calculate the difference between monthlyinstlamount_674A and monthlyinstlamount_332A. A larger difference could be relevant for risk assessment.
                (pl.col('monthlyinstlamount_332A').sum().fill_null(0.0) - pl.col('monthlyinstlamount_674A').sum().fill_null(0.0)).alias('monthlyinstlamount_332A_674A_diff'),

                # Instalment Coverage Ratio: Divide the total amount paid (monthlyinstlamount_674A) by the total amount due (monthlyinstlamount_332A). A higher ratio suggests better payment behavior.
                (pl.col('monthlyinstlamount_332A').sum().fill_null(0.0) / pl.col('monthlyinstlamount_674A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('monthlyinstlamount_332A_674A_ratio'),

                # Instalment Stability: Compare the number of instalments for active contracts with the average number of instalments for closed contracts. A significant deviation might indicate instability.
                (pl.col('numberofinstls_320L').sum().fill_null(0.0) / pl.col('numberofinstls_229L').sum().fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).alias('numberofinstls_320L_229L_sum_ratio'),
                (pl.col('numberofinstls_320L').mean().fill_null(0.0) / pl.col('numberofinstls_229L').mean().fill_null(0.0)).replace(float("inf"),0.0).fill_nan(0.0).alias('numberofinstls_320L_229L_mean_ratio'),

                # the ratio of the actual number of outstanding instalments 'numberofoutstandinstls_59L' to the total number of instalments 'numberofinstls_320L' for active contracts.
                (pl.col('numberofoutstandinstls_59L').sum().fill_null(0.0) / pl.col('numberofinstls_320L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoutstandinstls_59L_numberofinstls_320L_ratio'),
                # the ratio of the actual number of outstanding instalments 'numberofoutstandinstls_520L' to the total number of instalments 'numberofinstls_229L' for closed contracts.
                (pl.col('numberofoutstandinstls_520L').sum().fill_null(0.0) / pl.col('numberofinstls_229L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoutstandinstls_520L_numberofinstls_229L_ratio'),

                # Ratio of numberofoverdueinstlmax_1039L to numberofinstls_320L for active contract
                (pl.col('numberofoverdueinstlmax_1039L').sum().fill_null(0.0) / pl.col('numberofinstls_320L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoverdueinstlmax_1039L_numberofinstls_320L_ratio'),
                # Ratio of numberofoverdueinstlmax_1039L to numberofinstls_229L for closed contract
                (pl.col('numberofoverdueinstlmax_1151L').sum().fill_null(0.0) / pl.col('numberofinstls_229L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoverdueinstlmax_1151L_numberofinstls_229L_ratio'),

                # the ratio of the actual number of overdue instalments 'numberofoverdueinstls_725L' to the total number of instalments 'numberofinstls_320L' for active contracts.
                (pl.col('numberofoverdueinstls_725L').sum().fill_null(0.0) / pl.col('numberofinstls_320L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoverdueinstls_725L_numberofinstls_320L_ratio'),
                # the ratio of the actual number of overdue instalments 'numberofoverdueinstls_834L' to the total number of instalments 'numberofinstls_229L' for closed contracts.
                (pl.col('numberofoverdueinstls_834L').sum().fill_null(0.0) / pl.col('numberofinstls_229L').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('numberofoverdueinstls_834L_numberofinstls_229L_ratio'),

                # Ratio of outstanding amount outstandingamount_354A to credit limit credlmt_230A for closed contracts
                (pl.col('outstandingamount_354A').sum().fill_null(0.0) / pl.col('credlmt_230A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('outstandingamount_354A_credlmt_230A_ratio'),
                # Ratio of outstanding amount outstandingamount_362A to credit limit credlmt_935A for active contracts
                (pl.col('outstandingamount_362A').sum().fill_null(0.0) / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('outstandingamount_362A_credlmt_935A_ratio'),

                # Ratio of overdue amount overdueamount_31A to outstanding amount outstandingamount_354A for closed contracts
                (pl.col('overdueamount_31A').sum().fill_null(0.0) / pl.col('outstandingamount_354A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('overdueamount_31A_outstandingamount_354A_ratio'),
                # Ratio of overdue amount overdueamount_659A to outstanding amount outstandingamount_362A for active contracts
                (pl.col('overdueamount_659A').sum().fill_null(0.0) / pl.col('outstandingamount_362A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('overdueamount_659A_outstandingamount_362A_ratio'),

                # Residual Ratio: Compute the ratio between residualamount_856A and residualamount_488A.
                (pl.col('residualamount_856A').sum().fill_null(0.0) / pl.col('residualamount_488A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('residualamount_856A_488A_ratio'),
                # Normalized Residual Amounts: Calculate the normalized residual amounts by dividing each residual amount by the credit limit
                (pl.col('residualamount_856A').sum().fill_null(0.0) / pl.col('credlmt_935A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('residualamount_856A_credlmt_935A_ratio'),
                (pl.col('residualamount_488A').sum().fill_null(0.0) / pl.col('credlmt_230A').sum().fill_null(0.0)).replace(float("inf"), 0.0).fill_nan(0.0).alias('residualamount_488A_credlmt_230A_ratio'),

                # Contract Status Proportion: Calculate the proportion of active contracts (totalamount_996A) relative to the total (totalamount_996A + totalamount_6A) contracts.
                (pl.col('totalamount_6A').sum().fill_null(0.0) / (pl.col('totalamount_996A').sum().fill_null(0.0) + pl.col('totalamount_6A').sum().fill_null(0.0))).replace(float("inf"), 0.0).fill_nan(0.0).alias('totalamount_6A_totalamount_996A_ratio'),


                # Durations: mean, max,min value of durations of closed contracts ('dateofcredend_353D','dateofcredstart_181D')
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().mean().fill_null(0.0).alias('dateofcredend_353D_dateofcredstart_181D_diff_mean'),
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().max().fill_null(0.0).alias('dateofcredend_353D_dateofcredstart_181D_diff_max'),
                (pl.col("dateofcredend_353D") - pl.col("dateofcredstart_181D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_dateofcredstart_181D_diff_min'),

                # Durations: mean, max,min value of durations of active contracts ('dateofcredend_289D','dateofcredstart_739D')
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().mean().fill_null(0.0).alias('dateofcredend_289D_dateofcredstart_739D_diff_mean'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().max().fill_null(0.0).alias('dateofcredend_289D_dateofcredstart_739D_diff_max'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_289D_dateofcredstart_739D_diff_min'),
                (pl.col("dateofcredend_289D") - pl.col("dateofcredstart_739D")).dt.total_days().std().fill_null(0.0).alias('dateofcredend_289D_dateofcredstart_739D_diff_std'),

                # Difference between dateofcredend_353D and dateofrealrepmt_138D
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().mean().fill_null(0.0).alias('dateofcredend_353D_dateofrealrepmt_138D_diff'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().max().fill_null(0.0).alias('dateofcredend_353D_dateofrealrepmt_138D_max'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_dateofrealrepmt_138D_min'),
                (pl.col("dateofcredend_353D") - pl.col("dateofrealrepmt_138D")).dt.total_days().std().fill_null(0.0).alias('dateofcredend_353D_dateofrealrepmt_138D_std'),

                # Last updates:
                pl.col('lastupdate_1112D').max().alias('lastupdate_1112D_max'),
                pl.col('lastupdate_388D').max().alias('lastupdate_388D_max'),
                pl.col('lastupdate_1112D').min().alias('lastupdate_1112D_min'), # Contracts without long time update?
                pl.col('lastupdate_388D').min().alias('lastupdate_388D_min'),   # Contracts without long time update?

                # Latest date with maximum number of overdue instl (numberofoverdueinstlmaxdat_148D) and (numberofoverdueinstlmaxdat_641D)
                pl.col('numberofoverdueinstlmaxdat_148D').max().alias('numberofoverdueinstlmaxdat_148D_max'),
                pl.col('numberofoverdueinstlmaxdat_641D').max().alias('numberofoverdueinstlmaxdat_641D_max'),

                # remaining time of max overdue installments date till contract end
                (pl.col("dateofcredend_353D") - pl.col("numberofoverdueinstlmaxdat_148D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_numberofoverdueinstlmaxdat_148D_diff'),
                (pl.col("dateofcredend_289D") - pl.col("numberofoverdueinstlmaxdat_641D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_289D_numberofoverdueinstlmaxdat_641D_diff'),

                # Latest date with maximal overdue amount (overdueamountmax2date_1002D) and (overdueamountmax2date_1142D)
                pl.col('overdueamountmax2date_1002D').max().alias('overdueamountmax2date_1002D_max'),
                pl.col('overdueamountmax2date_1142D').max().alias('overdueamountmax2date_1142D_max'),
                
                # remaining time of max overdue amount date till contract end
                (pl.col("dateofcredend_353D") - pl.col("overdueamountmax2date_1002D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_overdueamountmax2date_1002D_diff'),
                (pl.col("dateofcredend_289D") - pl.col("overdueamountmax2date_1142D")).dt.total_days().min().fill_null(0.0).alias('dateofcredend_289D_overdueamountmax2date_1142D_diff'),

                # Date from str
                (pl.date(pl.col("overdueamountmaxdateyear_994T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_284T").cast(pl.Float64), 1).max()).alias("overdueamountmaxdate_994T_284T_fromstr_last"),
                (pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1).max()).alias("overdueamountmaxdate_2T_365T_fromstr_last"),

                # remaining time of max overdue amount date till contract end (from str version)
                (pl.col("dateofcredend_353D") - pl.date(pl.col("overdueamountmaxdateyear_994T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_284T").cast(pl.Float64), 1)).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_overdueamountmaxdate_994T_284T_diff'),
                (pl.col("dateofcredend_289D") - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().min().fill_null(0.0).alias('dateofcredend_289D_overdueamountmaxdate_2T_365T_diff'),
                # Assuming refreshdate_3813885D is current date, computing how much time ago this max overdue amount happend
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().min().alias("overdueamountmaxdate_2T_365T_refreshed_last"),
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("overdueamountmaxdateyear_2T").cast(pl.Float64),pl.col("overdueamountmaxdatemonth_365T").cast(pl.Float64), 1)).dt.total_days().mean().alias("overdueamountmaxdate_2T_365T_refreshed_mean"),

                # Date from str
                (pl.date(pl.col("dpdmaxdateyear_896T").cast(pl.Float64),pl.col("dpdmaxdatemonth_442T").cast(pl.Float64), 1).max()).alias("dpdmaxdate_896T_442T_fromstr_last"),
                (pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1).max()).alias("dpdmaxdate_596T_89T_fromstr_last"),

                 # remaining time of max dpd date till contract end (from str version)
                (pl.col("dateofcredend_353D") - pl.date(pl.col("dpdmaxdateyear_896T").cast(pl.Float64),pl.col("dpdmaxdatemonth_442T").cast(pl.Float64), 1)).dt.total_days().min().fill_null(0.0).alias('dateofcredend_353D_dpdmaxdate_896T_442T_diff'),
                (pl.col("dateofcredend_289D") - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().min().fill_null(0.0).alias('dateofcredend_289D_dpdmaxdate_596T_89T_diff'),
                # Assuming refreshdate_3813885D is current date, computing how much time ago this max overdue amount happend
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().min().alias("dpdmaxdateyear_596T_89T_refreshed_last"),
                (pl.col('refreshdate_3813885D') - pl.date(pl.col("dpdmaxdateyear_596T").cast(pl.Float64),pl.col("dpdmaxdatemonth_89T").cast(pl.Float64), 1)).dt.total_days().mean().alias("dpdmaxdateyear_596T_89T_refreshed_mean"),

                # Refresh date info
                pl.col('refreshdate_3813885D').min().alias('refreshdate_3813885D_min'),
                pl.col('refreshdate_3813885D').max().alias('refreshdate_3813885D_max'),
                pl.col('refreshdate_3813885D').mean().alias('refreshdate_3813885D_mean'),

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

            )

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
        
        # Relations between data tables for concatination of depth 1 and 2 after aggregation of depth 2
        data_relations_by_depth = {
            'applprev_1': 'applprev_2',
            'person_1': 'person_2',
            'credit_bureau_a_1': 'credit_bureau_a_2',
            'credit_bureau_b_1': 'credit_bureau_b_2'
        }


        # Reimplementation of reading and processing logic to benefit from lazy frames

        # Read the parquet files, concat, process, aggregate and join them in chains
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
            .join(query_credit_bureau_b_2, on=["case_id", "num_group1"], how='outer')
            .collect()
            .pipe(self.aggregate_depth_1, 'credit_bureau_b_1')
            .lazy()
        )

        howtojoin = 'left' if self.data_type=='test' else 'inner'
        query_base = (
            pl.read_parquet(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_base.parquet')
            .lazy()
            .join(query_credit_bureau_b_1, on="case_id", how=howtojoin) ## left, inner
            .collect()
        )


        # Step 2: credit_bureau_b_2 -> credit_bureau_b_1 -> base
        dataframes_credit_bureau_a_2 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_a_2*.parquet')):
            if ifile>1: continue
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
        query_credit_bureau_a_2 = pl.concat(dataframes_credit_bureau_a_2, how='vertical_relaxed')

        dataframes_credit_bureau_a_1 = []
        for ifile, file in enumerate(glob.glob(f'{self.data_path}parquet_files/{self.data_type}/{self.data_type}_credit_bureau_a_1*.parquet')):
            if ifile>1: continue
            q = (
                pl.read_parquet(file)
                .lazy()
                .pipe(self.set_table_dtypes)
                .pipe(self.encode_categorical_columns, 'credit_bureau_a_2')
            )
            dataframes_credit_bureau_a_1.append(q.collect())

        # Concat the dataframes
        query_credit_bureau_a_1 = pl.concat(dataframes_credit_bureau_a_1, how='vertical_relaxed')


        
        return query_credit_bureau_a_1 # query_base

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