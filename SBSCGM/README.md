# A Similarity-Based Self-Construct Graph Model for Predicting Sepsis Patient Mortality Using Graph Neural Networks

## 📌 Overview

Sepsis is a leading cause of mortality in Intensive Care Units (ICUs), requiring early detection and accurate predictions for timely interventions. Traditional models struggle to capture the complex relationships between patient demographics, diagnoses, treatments, and lab results due to the heterogeneous nature of Electronic Health Records (EHRs).

This research introduces **Similarity-Based Self-Construct Graph Model (SBSCGM)**, a novel **Graph Neural Network (GNN)-based framework** that dynamically constructs patient relationships for **sepsis mortality prediction**. Leveraging the **MIMIC-III dataset**, our approach represents **patients as nodes**, dynamically forming **edges based on feature-driven and structural similarity metrics** to improve predictive accuracy.

## 🚀 Key Features

- **Dynamic Graph Construction**: Builds patient similarity graphs using clinical features.
- **Graph Neural Networks (GNNs)**: Utilizes **Graph Convolutional Networks (GCNs)** and **Graph Attention Networks (GATs)** for prediction.
- **Advanced Embedding Techniques**: Encodes patient vectors in a shared space for accurate modeling.
- **Improved Sepsis Mortality Prediction**: Achieves **higher accuracy, precision, and interpretability** compared to traditional models.

## 📂 Dataset

This project uses the **MIMIC-III Clinical Database**, which contains **de-identified ICU patient records**. The dataset includes:
- **46,520** unique patients  
- **15,691** unique diagnoses  
- **5,854** sepsis-related mortalities  
- **1,633** sepsis patients identified based on primary diagnosis  

🔗 **Access the dataset**: [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/1.4/)

## 🛠 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/mukesh16/Expt_GNNs_EHR.git
cd Expt_GNNs_EHR
```

### 2️⃣ **Create a Virtual Environment (Recommended)**
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Database Setup**
Ensure you have PostgreSQL installed and set up to query the MIMIC-III dataset using Google BigQuery.

### 5️⃣ **Run the Model**
To train and evaluate the GNN model:
```bash
python train_model.py
```

## 🏗 Model Architecture

The proposed SBSCGM framework follows these steps:
1. **Data Preprocessing**: Cleans and integrates patient data from MIMIC-III.
2. **Graph Construction**: Forms a dynamic patient graph based on feature and structural similarity.
3. **GNN Training**: Applies GCNs and GATs to learn node representations.
4. **Sepsis Mortality Prediction**: Evaluates performance using AUC-ROC, Precision, Recall, F1-score.

## 📊 Experimental Results

The SBSCGM model outperforms traditional models across key evaluation metrics:

| Model                    | AUC-ROC | Accuracy | Precision | Recall |
|--------------------------|---------|----------|-----------|--------|
| Logistic Regression       | 0.78    | 76.3%    | 72.5%     | 68.2%  |
| Random Forest             | 0.81    | 78.6%    | 75.1%     | 70.8%  |
| Graph Neural Networks (SBSCGM) | 0.91    | 86.2%    | 83.5%     | 80.7%  |

## 🔬 Research Paper

This work is part of an ongoing research project in predictive healthcare analytics. The paper draft and detailed methodology can be found in: 📄 [Research Paper](#)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an issue for bug reports or feature suggestions.
- Submit a pull request for enhancements.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Mukesh Kumar Sahu  
MTech in Data Science & Engineering, NIT Silchar  
🚀 Target Role: Data Scientist specializing in AI & ML

- LinkedIn: [Mukesh Kumar Sahu](https://www.linkedin.com/in/mukesh-kumar-sahu/)
- Email: [sahumukeshkumar16@gmail.com](mailto:sahumukeshkumar16@gmail.com) | [mukesh_pg_23@cse.nits.ac.in](mailto:mukesh_pg_23@cse.nits.ac.in)
- ORCID iD: [0009-0005-4007-7648](https://orcid.org/0009-0005-4007-7648)
- GitHub: [mukesh16](https://github.com/mukesh16)
