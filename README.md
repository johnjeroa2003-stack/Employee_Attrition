# Employee Attrition Analysis and Prediction

This project skeleton was auto-generated. It includes scripts for preprocessing, model training, a Streamlit app, and a starter notebook.

**Dataset**
- `data/Employee_Attrition.csv` — dataset (copied/created)

**Quick Start (VS Code)**
1. Create & activate a virtual environment:
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run preprocessing:
   ```
   python scripts/data_preprocessing.py
   ```
4. Train model:
   ```
   python scripts/model_training.py
   ```
5. Run Streamlit app:
   ```
   streamlit run app/attrition_dashboard.py
   ```

**Files**
- `scripts/data_preprocessing.py` — cleaning & encoding.
- `scripts/model_training.py` — train RandomForest, save model and scaler.
- `notebooks/EDA_Attrition_Analysis.ipynb` — starter EDA notebook.
- `app/attrition_dashboard.py` — Streamlit app (dashboard + prediction).
