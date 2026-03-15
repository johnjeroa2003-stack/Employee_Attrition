import pandas as pd

print("Starting Data Preprocessing")

# Load Dataset
df = pd.read_csv("../data/Employee-Attrition - Employee-Attrition.csv")

print("Dataset Loaded")

# =========================
# Feature Engineering
# =========================

# Tenure Category
df["TenureCategory"] = pd.cut(
    df["YearsAtCompany"],
    bins=[0,3,7,15,40],
    labels=["New","Junior","Mid","Senior"]
)

# Income Level
df["IncomeLevel"] = pd.cut(
    df["MonthlyIncome"],
    bins=[1000,5000,10000,20000],
    labels=["Low","Medium","High"]
)

# Engagement Score
df["EngagementScore"] = (
    df["JobSatisfaction"] +
    df["EnvironmentSatisfaction"] +
    df["RelationshipSatisfaction"] +
    df["WorkLifeBalance"]
)

# Promotion Delay
df["PromotionDelay"] = (
    df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
)

print("Feature Engineering Completed")

# =========================
# Drop unnecessary columns
# =========================

df = df.drop(["EmployeeCount","EmployeeNumber","StandardHours","Over18"], axis=1)

print("Dropped unnecessary columns")

# =========================
# Convert Attrition column
# =========================

df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

print("Converted Attrition column")

# =========================
# Encode categorical columns
# =========================

df = pd.get_dummies(df, drop_first=True)

print("Categorical variables encoded")

# =========================
# Save Clean Dataset
# =========================

df.to_csv("../data/cleaned_employee_data.csv", index=False)

print("Cleaned dataset saved")

print("Final dataset shape:", df.shape)
