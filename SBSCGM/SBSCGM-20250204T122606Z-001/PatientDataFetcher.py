from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from google.colab import auth

auth.authenticate_user()


class PatientDataFetcher:
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
                'diagnoses_info': executor.submit(self.fetch_diagnoses_info, subject_filter),
                'procedures_info': executor.submit(self.fetch_procedures_info, subject_filter),
                'prescriptions_info': executor.submit(self.fetch_prescriptions_info, subject_filter),
                'lab_events_info': executor.submit(self.fetch_lab_events_info, subject_filter),
                'chart_events_info': executor.submit(self.fetch_chart_events_info, subject_filter),
                'input_events_info': executor.submit(self.fetch_input_events_info, subject_filter),
                'output_events_info': executor.submit(self.fetch_output_events_info, subject_filter),
            }
        
            self.data = {key: future.result() for key, future in futures.items()}
        return self.data

    def fetch_patient_info(self, filter_condition):
        return self.execute_query("patients", filter_condition)

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

    def fetch_chart_events_info(self, filter_condition):
        return self.execute_query("chartevents", filter_condition)

    def fetch_input_events_info(self, filter_condition):
        return self.execute_query("inputevents_cv", filter_condition)

    def fetch_output_events_info(self, filter_condition):
        return self.execute_query("outputevents", filter_condition)

    def execute_query(self, table_name, filter_condition):
        query = f"SELECT * FROM `{self.dataset_name}.{table_name}` {filter_condition}"
        print(f"Executing query: {query}")
        try:
            df = self.client.query(query).to_dataframe()
            print(f"Fetched {len(df)} records from {table_name}.")
            return df
        except Exception as e:
            print(f"Error fetching {table_name} data: {e}")
            return pd.DataFrame()
        

class PatientDataFrame(PatientDataFetcher):
    def __init__(self, project_id, dataset_name):
        super().__init__(project_id, dataset_name)
    
    def prepare_dataframe(self):
        if not self.data:
            raise ValueError("No data available. Fetch data first.")
        
        df_list = [df for df in self.data.values() if not df.empty]
        if df_list:
            self.df = pd.concat(df_list, axis=1, join='outer')
            self.df.drop_duplicates(inplace=True)
        
        return self.df

