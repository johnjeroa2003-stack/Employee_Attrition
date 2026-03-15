import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------- Load Model --------------------

model = pickle.load(open("model/attrition_model.pkl", "rb"))

with open("model/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# -------------------- Load Dataset --------------------

df = pd.read_csv("data/cleaned_employee_data.csv")
st.write("Dataset Columns:", df.columns)

# Clean column names (removes hidden spaces)
df.columns = df.columns.str.strip()

st.write("Columns in dataset:", df.columns)

# -------------------- Title --------------------

st.title("Employee Attrition Prediction System")
st.write("Enter employee details to predict attrition risk")

# -------------------- User Inputs --------------------

age = st.number_input("Age", 18, 65)
monthly_income = st.number_input("Monthly Income", 1000, 50000)
years_at_company = st.number_input("Years At Company", 0, 40)

# -------------------- Prediction --------------------

if st.button("Predict"):

    input_dict = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns correctly
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Employee likely to leave")
    else:
        st.success("✅ Employee likely to stay")

# -------------------- Charts --------------------

st.subheader("Attrition Distribution")

if "Attrition" in df.columns:
    attrition_counts = df["Attrition"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(attrition_counts.index, attrition_counts.values)

    st.pyplot(fig)
else:
    st.warning("Attrition column not found in dataset")

# -------------------- Department Chart --------------------

st.subheader("Attrition by Department")

dept_columns = [col for col in df.columns if "Department_" in col]

if len(dept_columns) > 0:

    dept_chart = df[dept_columns].sum()

    fig, ax = plt.subplots()
    dept_chart.plot(kind="bar", ax=ax)

    st.pyplot(fig)

else:
    st.warning("No Department information found in dataset")
# -------------------- Income Distribution --------------------

st.subheader("Monthly Income Distribution")

if "MonthlyIncome" in df.columns:

    fig, ax = plt.subplots()
    ax.hist(df["MonthlyIncome"], bins=30)

    st.pyplot(fig)

else:
    st.warning("MonthlyIncome column not found")
