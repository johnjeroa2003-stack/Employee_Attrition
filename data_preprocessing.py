import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/Employee_Attrition.csv")
OUT_PATH = Path("data/cleaned_employee_data.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    print("Initial shape:", df.shape)
    # Drop mostly-constant/unnecessary columns if present
    drop_cols = [c for c in ['EmployeeCount','Over18','StandardHours','EmployeeNumber'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    # Basic missing handling
    print("Missing values per column:\n", df.isnull().sum())
    # For simplicity, drop rows with missing values
    df = df.dropna().copy()
    # Encode target if it's Yes/No
    if df['Attrition'].dtype == object:
        df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Attrition']
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df.to_csv(OUT_PATH, index=False)
    print("Cleaned data saved to", OUT_PATH)

if __name__ == '__main__':
    main()
