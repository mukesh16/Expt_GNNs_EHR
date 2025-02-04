import pandas as pd
import numpy as np
from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_networkx
from google.colab import auth
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

auth.authenticate_user()

class PatientDataFetcher:
    def __init__(self, project_id, dataset_name):
        self.client = bigquery.Client(project=project_id)
        self.dataset_name = dataset_name
        self.data = {}
        self.df = pd.DataFrame()

    def fetch_data(self, subject_ids):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'patient_info': executor.submit(self.fetch_patient_info, subject_ids),
                'admissions_info': executor.submit(self.fetch_admissions_info, subject_ids),
                'icu_info': executor.submit(self.fetch_icu_info, subject_ids),
                'diagnoses_info': executor.submit(self.fetch_diagnoses_info, subject_ids),
                'procedures_info': executor.submit(self.fetch_procedures_info, subject_ids),
                'prescriptions_info': executor.submit(self.fetch_prescriptions_info, subject_ids),
                'lab_events_info': executor.submit(self.fetch_lab_events_info, subject_ids),
                'chart_events_info': executor.submit(self.fetch_chart_events_info, subject_ids),
                'input_events_info': executor.submit(self.fetch_input_events_info, subject_ids),
                'output_events_info': executor.submit(self.fetch_output_events_info, subject_ids),
            }
        
            self.data = {key: future.result() for key, future in futures.items()}
        return self.data

    def get_dataframe(self, data_dict=None, chunk_size=1000):
        if data_dict is None:
            data_dict = self.data

        df_list = []
        seen_columns = set()
        
        first_iteration = True
        for key, data in data_dict.items():
            data.columns = [col.lower() for col in data.columns]

            columns_to_keep = [col for col in data.columns if col not in ['row_id', 'subject_id', 'hadm_id', 'icustay_id', 'itemid', 'cgid']]
            columns_to_key = [col for col in data.columns if col in ['subject_id', 'hadm_id', 'icustay_id', 'itemid', 'cgid']]
            seen_columns.update(columns_to_key)

            # Chunk processing
            chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            for chunk in chunks:
                if first_iteration:
                    df_list.append(chunk.drop(columns=['row_id'], errors='ignore').copy())
                else:
                    common_keys = list(set(columns_to_key) & set(df_list[0].columns) & set(chunk.columns))
                    if common_keys:
                        for i, df in enumerate(df_list):
                            df_list[i] = pd.merge(df, chunk.drop(columns=['row_id'], errors='ignore'), on=common_keys, how='left', suffixes=('', f'_{key}'))
                    else:
                        print(f"Skipping merge for {key} due to missing key columns.")

                first_iteration = False

        # Combine chunks
        df = pd.concat(df_list, ignore_index=True)

        # Memory Optimization
        df = self.optimize_dataframe_memory(df)
        
        df.dropna(axis=1, how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        self.df = df.copy()
        
        return self.df

    def fetch_patient_info(self, subject_ids):
        query = f"""
        SELECT * FROM `{self.dataset_name}.patients`
        """
        return self.client.query(query).to_dataframe()

    def fetch_admissions_info(self, subject_ids):
        query = f"""
        SELECT * FROM `{self.dataset_name}.admissions`
        """
        return self.client.query(query).to_dataframe()

    def fetch_icu_info(self, subject_ids):
        query = f"""
        SELECT * FROM `{self.dataset_name}.icustays`
        """
        return self.client.query(query).to_dataframe()

    def fetch_diagnoses_info(self, subject_ids):
        query = f"""
        SELECT d_icd.*, dicd.long_title 
        FROM `{self.dataset_name}.diagnoses_icd` d_icd
        LEFT JOIN `{self.dataset_name}.d_icd_diagnoses` dicd 
        ON d_icd.icd9_code = dicd.icd9_code
        """
        return self.client.query(query).to_dataframe()

    def fetch_procedures_info(self, subject_ids):
        query = f"""
        SELECT proc.*, dproc.long_title 
        FROM `{self.dataset_name}.procedures_icd` proc
        LEFT JOIN `{self.dataset_name}.d_icd_procedures` dproc 
        ON proc.icd9_code = dproc.icd9_code
        """
        return self.client.query(query).to_dataframe()

    def fetch_prescriptions_info(self, subject_ids):
        query = f"""
        SELECT * FROM `{self.dataset_name}.prescriptions`
        """
        return self.client.query(query).to_dataframe()

    def fetch_lab_events_info(self, subject_ids):
        query = f"""
        SELECT l.*, dl.label 
        FROM `{self.dataset_name}.labevents` l
        LEFT JOIN `{self.dataset_name}.d_labitems` dl 
        ON l.itemid = dl.itemid
        """
        return self.client.query(query).to_dataframe()

    def fetch_chart_events_info(self, subject_ids):
        query = f"""
        SELECT ce.*, dce.label 
        FROM `{self.dataset_name}.chartevents` ce
        LEFT JOIN `{self.dataset_name}.d_items` dce 
        ON ce.itemid = dce.itemid
        """
        return self.client.query(query).to_dataframe()

    def fetch_input_events_info(self, subject_ids):
        query = f"""
        SELECT ie.*, die.label 
        FROM `{self.dataset_name}.inputevents_cv` ie
        LEFT JOIN `{self.dataset_name}.d_items` die 
        ON ie.itemid = die.itemid
        """
        return self.client.query(query).to_dataframe()

    def fetch_output_events_info(self, subject_ids):
        query = f"""
        SELECT oe.*, doe.label 
        FROM `{self.dataset_name}.outputevents` oe
        LEFT JOIN `{self.dataset_name}.d_items` doe 
        ON oe.itemid = doe.itemid
        """
        return self.client.query(query).to_dataframe()


class DataPreprocessor:
    @staticmethod
    def preprocess_data(df):
        # Identify columns by data type
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols.remove('expire_flag')
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Convert datetime columns to numeric features
        for col in datetime_cols:
            if col in df.columns:
                df[col + '_year'] = df[col].dt.year
                df[col + '_month'] = df[col].dt.month
                df[col + '_day'] = df[col].dt.day
                df[col + '_hour'] = df[col].dt.hour
                df.drop(columns=[col], inplace=True)

        # Fill missing values
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[categorical_cols] = df[categorical_cols].fillna('missing')

        # Separate categorical columns by cardinality
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 10]
        low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 10]

        # Label Encoding for high cardinality columns
        for col in high_cardinality_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # One-Hot Encoding for low cardinality columns
        df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True, sparse=True)
        
        df.dropna(axis=1, how='all', inplace=True)
        df.drop_duplicates(inplace=True)

        return df.reset_index(drop=True)


class GraphConstructor:
    @staticmethod
    def construct_graph(df):
        G = nx.Graph()

        # Define primary key columns and relevant features
        primary_key = 'subject_id'
        feature_columns = [col for col in df.columns if not col.endswith('_missing') and col not in [
            'hadm_id', 'icustay_id', 'itemid', 'cgid', 'orderid', 'iserror', primary_key
        ]]  # All columns except those related to relationships

        # Add patient nodes with their features
        for _, row in df.iterrows():
            G.add_node(row[primary_key],
                    node_type='patient',
                    **{col: row[col] for col in feature_columns if col in row})

        # Define relationships based on specific logic
        relationships = {
            'hadm_id': ['admittime_year', 'dischtime_year'],
            'icustay_id': ['intime_year', 'outtime_year'],
            'itemid': ['charttime_year'],
            'cgid': ['storetime_year'],
            'orderid': ['startdate_year', 'enddate_year'],
            'iserror': []
        }

        for rel, time_cols in relationships.items():
            if rel in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[rel]):
                        entity_id = row[rel]
                        if not G.has_node(entity_id):
                            # Add default attributes to ensure consistency
                            G.add_node(entity_id, node_type=rel, **{col: 0 for col in feature_columns})

                        edge_attrs = {f'time_{i}': row[time_col] for i, time_col in enumerate(time_cols) if time_col in row}
                        G.add_edge(row[primary_key], entity_id, edge_type=rel, **edge_attrs)

        return G

    @staticmethod
    def plot_graph(G):
        pos = nx.spring_layout(G)
        unique_nodes = set(G.nodes())
        node_colors = ['lightblue' if isinstance(node, int) else 'lightgreen' for node in G.nodes()]
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, font_size=10, font_weight='bold', edge_color='gray')
        
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Primary Nodes (e.g., patients)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Related Nodes')
        ]
        plt.legend(handles=legend_handles, loc='best')
        plt.title('Graph Visualization')
        plt.show()

    @staticmethod
    def prepare_data_for_gcn(G, df):
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].astype(np.int64) // 10**9  # Convert to Unix timestamp in seconds
        
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)
        
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # Convert to numeric, fill NaNs with 0
            except ValueError:
                print(f"Column '{col}' could not be converted to numeric and will be dropped.")
                df = df.drop(columns=[col])  # Drop the column if conversion fails
                
        # Convert relevant columns to numeric format individually
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Attempt to convert to numeric if elements are numbers
                    df[col] = pd.to_numeric(df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) > 0 else x), errors='coerce')
                except (TypeError, ValueError):
                    print(f"Column '{col}' could not be converted to numeric and will be dropped.")
                    df = df.drop(columns=[col])  # Drop the column if conversion fails
    
        # Ensure all remaining columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        X = torch.tensor(df.values.astype(np.float32), dtype=torch.float32)
        A = nx.to_scipy_sparse_array(G)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        data = Data(x=X, edge_index=edge_index)
        
        if 'expire_flag' in df.columns:
            data.y = torch.tensor(df['expire_flag'].values, dtype=torch.float32) 
        else:
            raise ValueError("Target column 'expire_flag' not found in the DataFrame.")

        return data


class GNNTrainer:
    def __init__(self, model, data, train_mask, val_mask):
        self.model = model
        self.data = data
        self.train_mask = train_mask
        self.val_mask = val_mask

    def train(self, epochs, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data)
            # loss = F.nll_loss(out[self.train_mask], self.data.y[self.train_mask])
            loss = F.nll_loss(out[self.train_mask], self.data.y[self.train_mask].long())
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(self.data)
                val_loss = F.nll_loss(val_out[self.val_mask], self.data.y[self.val_mask].long())
            self.model.train()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            pred = out[test_mask].max(dim=1)[1]
            acc = pred.eq(self.data.y[test_mask]).sum().item() / test_mask.sum().item()
        return acc, pred, self.data.y[test_mask]

    @staticmethod
    def shuffle_and_stratify_split_data(data, train_ratio=0.8, val_ratio=0.1):
        train_idx, test_idx = train_test_split(range(data.num_nodes), train_size=train_ratio, stratify=data.y)
        val_idx, test_idx = train_test_split(test_idx, train_size=val_ratio/(1-train_ratio), stratify=data.y[test_idx])
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        return train_mask, val_mask, test_mask


class CustomGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.5):
        super(CustomGAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=heads)
        self.conv2 = GATConv(8 * heads, out_channels, heads=1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# # Example usage:
# # Assuming you have a project_id, dataset_name, and subject_ids

# project_id = 'your_project_id'
# dataset_name = 'your_dataset_name'
# subject_ids = [123, 456, 789]  # Example subject IDs

# fetcher = PatientDataFetcher(project_id, dataset_name)
# data_dict = fetcher.fetch_data(subject_ids)
# df = fetcher.get_dataframe(data_dict)

# preprocessor = DataPreprocessor()
# preprocessed_df = preprocessor.preprocess_data(df)

# graph_constructor = GraphConstructor()
# G = graph_constructor.construct_graph(preprocessed_df)
# graph_constructor.plot_graph(G)
# data = graph_constructor.prepare_data_for_gcn(G, preprocessed_df)

# train_mask, val_mask, test_mask = GNNTrainer.shuffle_and_stratify_split_data(data)

# model = CustomGAT(in_channels=data.num_features, out_channels=2)
# trainer = GNNTrainer(model, data, train_mask, val_mask)
# trainer.train(epochs=100, lr=0.01)
# accuracy, predictions, targets = trainer.evaluate(test_mask)

# print(f'Accuracy: {accuracy}')
