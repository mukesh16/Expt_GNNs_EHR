from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

from google.colab import auth

auth.authenticate_user()


class SepsisPatientDataFetcher:
    def __init__(self, project_id, dataset_name):
        self.client = bigquery.Client(project=project_id)
        self.dataset_name = dataset_name
        self.data = {}
        self.df = pd.DataFrame()

    def fetch_admissions_for_sepsis(self):
        query = f"""
        SELECT DISTINCT subject_id FROM `{self.dataset_name}.admissions`
        WHERE diagnosis LIKE '%SEPSIS%'
        """
        print(f"Executing query: {query}")
        try:
            admissions_df = self.client.query(query).to_dataframe()
            unique_subjects = admissions_df['subject_id'].nunique()
            print(f"Fetched {unique_subjects} unique subject IDs related to sepsis.")
            return admissions_df
        except Exception as e:
            print(f"Error fetching admissions data: {e}")
            return pd.DataFrame()

    def fetch_data(self):
        admissions_df = self.fetch_admissions_for_sepsis()
        if admissions_df.empty:
            print("No sepsis-related admissions found.")
            return {}
        
        subject_ids = admissions_df['subject_id'].tolist()
        subject_filter = f"WHERE subject_id IN ({', '.join(map(str, subject_ids))})"
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'patient_info': executor.submit(self.fetch_patient_info, subject_filter),
                'icu_info': executor.submit(self.fetch_icu_info, subject_filter),
                'admissions_info': executor.submit(self.fetch_admissions_info, subject_filter),
                'diagnoses_info': executor.submit(self.fetch_diagnoses_info, subject_filter),
                'procedures_info': executor.submit(self.fetch_procedures_info, subject_filter),
                'prescriptions_info': executor.submit(self.fetch_prescriptions_info, subject_filter),
                'lab_events_info': executor.submit(self.fetch_lab_events_info, subject_filter),
                # 'chart_events_info': executor.submit(self.fetch_chart_events_info, subject_filter),
                # 'input_events_info': executor.submit(self.fetch_input_events_info, subject_filter),
                # 'output_events_info': executor.submit(self.fetch_output_events_info, subject_filter),
            }
        
            self.data = {key: future.result() for key, future in futures.items()}
        return self.data

    def fetch_patient_info(self, filter_condition):
        return self.execute_query("patients", filter_condition)
        
    def fetch_admissions_info(self, filter_condition):
        return self.execute_query("admissions", filter_condition)

    def fetch_icu_info(self, filter_condition):
        return self.execute_query("icustays", filter_condition)

    def fetch_diagnoses_info(self, filter_condition):
        return self.execute_query("diagnoses_icd", filter_condition)

    def fetch_procedures_info(self, filter_condition):
        return self.execute_query("procedures_icd", filter_condition)

    def fetch_prescriptions_info(self, filter_condition):
        return self.execute_query("prescriptions", filter_condition)

    def fetch_lab_events_info(self, filter_condition):
        return self.execute_query("labevents", filter_condition)

    # def fetch_chart_events_info(self, filter_condition):
    #     return self.execute_query("chartevents", filter_condition)

    # def fetch_input_events_info(self, filter_condition):
    #     return self.execute_query("inputevents_cv", filter_condition)

    # def fetch_output_events_info(self, filter_condition):
    #     return self.execute_query("outputevents", filter_condition)

    def execute_query(self, table_name, filter_condition):
        query = f"SELECT * EXCEPT(ROW_ID) FROM `{self.dataset_name}.{table_name}` {filter_condition}"
        print(f"Executing query: {query}")
        try:
            df = self.client.query(query).to_dataframe()
            print(f"Fetched {len(df)} records from {table_name}.")
            return df
        except Exception as e:
            print(f"Error fetching {table_name} data: {e}")
            return pd.DataFrame()
            

class SepsisPatientDataFrame(SepsisPatientDataFetcher):
    def __init__(self, project_id, dataset_name):
        super().__init__(project_id, dataset_name)
    
    def prepare_dataframe(self):
        self.fetch_data()
        df_list = [df for df in self.data.values() if not df.empty]
        
        # Drop 'ROW_ID' column if it exists
        for key in self.data.keys():
            if 'ROW_ID' in self.data[key].columns:
                self.data[key].drop(columns=['ROW_ID'], inplace=True)
        
        # Convert column names to lowercase
        for key in self.data.keys():
            self.data[key].columns = [col.lower() for col in self.data[key].columns]
        
        # Merge data on common columns
        self.df = self.merge_dataframes()

        # Preprocess the data
        return self.preprocess_dataframe()
    
    def merge_dataframes(self):
        keys_to_delete = ["chart_events_info", "input_events_info", "output_events_info"]
        for key in keys_to_delete:
            if key in self.data:
                del self.data[key]

        df_list = [df for df in self.data.values() if not df.empty]
        if not df_list:
            print("No data available for merging.")
            return pd.DataFrame()

        # # Identify common columns for merging
        # common_cols = set.intersection(*[set(df.columns) for df in df_list]) if df_list else set()
        # if not common_cols:
        #     print("No common columns found for merging. Returning concatenated DataFrame.")
        #     return pd.concat(df_list, axis=1, join='outer')

        # merge_col = 'subject_id' if 'subject_id' in common_cols else list(common_cols)[0]
        # df_list = [df for df in df_list if merge_col in df.columns]

        # if not df_list:
        #     print(f"No DataFrames contain the merge column {merge_col}. Returning empty DataFrame.")
        #     return pd.DataFrame()

        # merged_df = df_list[0]
        # for i, df in enumerate(df_list[1:], start=2):
        #     print(f"Merging {i}/{len(df_list)} DataFrames on {merge_col}")
        #     try:
        #         merged_df = merged_df.merge(df, on=merge_col, how='outer', suffixes=(f'_df{i-1}', f'_df{i}'))
        #     except Exception as e:
        #         print(f"Error merging DataFrame {i}: {e}")
        #         continue
        
        # return merged_df
        
        # Identify common columns across all DataFrames
        common_cols = set.intersection(*[set(df.columns) for df in df_list]) if df_list else set()
        if not common_cols:
            print("No common columns found for merging. Returning concatenated DataFrame.")
            return pd.concat(df_list, axis=1, join='inner')

        # Merge all DataFrames using all common columns
        merged_df = df_list[0]
        for i, df in enumerate(df_list[1:], start=2):
            print(f"Merging {i}/{len(df_list)} DataFrames on {common_cols}")
            merged_df = merged_df.merge(df, on=list(common_cols), how='inner', suffixes=(f'_df{i-1}', f'_df{i}'))
        
        return merged_df

    def preprocess_dataframe(self):
        if self.df.empty:
            print("No data available for preprocessing.")
            return self.df
        
        # Handling missing values
        self.df.fillna(0, inplace=True)
        
        # Identify columns to skip for encoding
        problematic_cols = ["value", "valueuom", "stopped"]
        categorical_cols = [col for col in self.df.select_dtypes(include=['object']).columns if col not in problematic_cols]

        # Encode categorical variables
        for col in categorical_cols:
            if self.df[col].ndim == 1:
                self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))
            else:
                print(f"Skipping column {col} as it is not 1D.")

        # Scale numerical variables
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[numeric_cols] = StandardScaler().fit_transform(self.df[numeric_cols])

        print("Data preprocessing completed.")
        return self.df
        
class SepsisPatientBigQueryFetcher:
    def __init__(self, project_id, dataset_name):
        self.client = bigquery.Client(project=project_id)
        self.dataset_name = dataset_name

    def fetch_sepsis_data(self):
        query = f"""
        WITH sepsis_patients AS (
            SELECT DISTINCT subject_id, hadm_id 
            FROM `{self.dataset_name}.admissions`
            WHERE LOWER(diagnosis) LIKE '%sepsis%'
        ),
        
        filtered_labs AS (
            SELECT * EXCEPT(ROW_ID) 
            FROM `{self.dataset_name}.labevents`
            WHERE subject_id IN (SELECT subject_id FROM sepsis_patients)
        ),
        
        filtered_prescriptions AS (
            SELECT * EXCEPT(ROW_ID) 
            FROM `{self.dataset_name}.prescriptions`
            WHERE subject_id IN (SELECT subject_id FROM sepsis_patients)
        ),
        
        filtered_diagnoses AS (
            SELECT * EXCEPT(ROW_ID)
            FROM `{self.dataset_name}.diagnoses_icd`
            WHERE subject_id IN (SELECT subject_id FROM sepsis_patients)
        ),
        
        filtered_procedures AS (
            SELECT * EXCEPT(ROW_ID)
            FROM `{self.dataset_name}.procedures_icd`
            WHERE subject_id IN (SELECT subject_id FROM sepsis_patients)
        )
        
        SELECT 
            p.SUBJECT_ID, p.GENDER, p.DOB, p.DOD, p.DOD_HOSP, p.DOD_SSN, p.EXPIRE_FLAG,
            a.HADM_ID, a.ADMISSION_LOCATION, a.ADMISSION_TYPE, a.ADMITTIME, a.DEATHTIME, a.HOSPITAL_EXPIRE_FLAG, a.RELIGION, 
            a.DISCHARGE_LOCATION, a.DISCHTIME, a.EDOUTTIME, a.EDREGTIME, a.ETHNICITY, a.INSURANCE, a.LANGUAGE, a.MARITAL_STATUS,
            icu.ICUSTAY_ID, icu.FIRST_CAREUNIT, icu.FIRST_WARDID, icu.INTIME, icu.OUTTIME, icu.LAST_CAREUNIT, icu.LAST_WARDID,
            d.ICD9_CODE, d.SEQ_NUM, proc.ICD9_CODE, proc.SEQ_NUM, pres.STARTDATE, pres.DRUG, pres.DRUG_TYPE,
            pres.DRUG_NAME_GENERIC, pres.DRUG_NAME_POE, pres.DOSE_UNIT_RX, pres.DOSE_VAL_RX, pres.ENDDATE, 
            pres.ROUTE, pres.FORMULARY_DRUG_CD, pres.GSN, pres.NDC, pres.PROD_STRENGTH, pres.FORM_UNIT_DISP, pres.FORM_VAL_DISP,
            lab.ITEMID, lab.VALUE, lab.VALUENUM, lab.VALUEUOM, lab.CHARTTIME, lab.FLAG
        FROM sepsis_patients sp
        LEFT JOIN `{self.dataset_name}.patients` p ON sp.subject_id = p.subject_id
        LEFT JOIN `{self.dataset_name}.admissions` a ON sp.hadm_id = a.hadm_id
        LEFT JOIN `{self.dataset_name}.icustays` icu ON sp.hadm_id = icu.hadm_id
        LEFT JOIN filtered_diagnoses d ON sp.hadm_id = d.hadm_id
        LEFT JOIN filtered_procedures proc ON sp.hadm_id = proc.hadm_id
        LEFT JOIN filtered_prescriptions pres ON sp.hadm_id = pres.hadm_id
        LEFT JOIN filtered_labs lab ON sp.hadm_id = lab.hadm_id;
        """
        print("Executing optimized BigQuery fetch for sepsis patient data...")
        try:
            df = self.client.query(query).to_dataframe()
            print(f"Fetched {len(df)} records from the BigQuery execution.")
            return df
        except Exception as e:
            print(f"Error executing BigQuery fetch: {e}")
            return pd.DataFrame()
        

# Example usage:
project_id = 'your_project_id'
dataset_name = 'mimiciii_clinical'

fetcher = SepsisPatientBigQueryFetcher(project_id, dataset_name)
df = fetcher.fetch_sepsis_data()
print(df.head())
