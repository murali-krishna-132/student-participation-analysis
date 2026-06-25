# ======================================================
# WEEK 2 - COHORT PARTICIPATION ANALYSIS
# Student Participation Dataset
# ======================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("dataset/Cleaned_Student_Dataset.csv")

print("="*50)
print("Dataset Shape")
print(df.shape)

print("="*50)
print(df.head())

print("="*50)
print(df.info())

# ======================================================
# Data Cleaning
# ======================================================

print("\nMissing Values\n")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Standardize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ","_")
      .str.replace("/","_")
)

print(df.columns)

# ======================================================
# Convert Date Columns
# ======================================================

date_columns = [
    "learner_signup_datetime",
    "date_of_birth",
    "entry_created_at",
    "apply_date",
    "opportunity_start_date",
    "opportunity_end_date"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ======================================================
# Handle Missing Values
# ======================================================

for col in df.select_dtypes(include="object"):
    df[col].fillna("Unknown", inplace=True)

for col in df.select_dtypes(include=["float64","int64"]):
    df[col].fillna(df[col].median(), inplace=True)

# ======================================================
# Feature Engineering
# ======================================================

df["signup_year"] = df["learner_signup_datetime"].dt.year

df["signup_month"] = df["learner_signup_datetime"].dt.month_name()

df["signup_day"] = df["learner_signup_datetime"].dt.day_name()

df["signup_hour"] = df["learner_signup_datetime"].dt.hour

df["application_delay"] = (
    df["apply_date"] -
    df["learner_signup_datetime"]
).dt.days

df["opportunity_duration"] = (
    df["opportunity_end_date"] -
    df["opportunity_start_date"]
).dt.days

print(df.head())

# Create Engagement Intensity Score

df["engagement_intensity"] = (
    df["application_delay"].fillna(0).abs() +
    df["opportunity_duration"].fillna(0)
)

# ======================================================
# Descriptive Statistics
# ======================================================

print("\nSummary Statistics\n")

print(df.describe(include="all"))

# ======================================================
# Status Distribution
# ======================================================

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    y="status_description",
    order=df["status_description"].value_counts().index
)

plt.title("Participation Status Distribution")

plt.xlabel("Count")

plt.ylabel("Status")

plt.tight_layout()

plt.savefig("status_distribution.png")

plt.show()

# ======================================================
# Opportunity Category
# ======================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="opportunity_category",
    order=df["opportunity_category"].value_counts().index
)

plt.title("Opportunity Category Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("opportunity_category.png")

plt.show()

# ======================================================
# Gender Distribution
# ======================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="gender"
)

plt.title("Gender Distribution")

plt.tight_layout()

plt.savefig("gender_distribution.png")

plt.show()

# ======================================================
# Top Countries
# ======================================================

plt.figure(figsize=(10,5))

top_country = df["country"].value_counts().head(10)

sns.barplot(
    x=top_country.values,
    y=top_country.index
)

plt.title("Top 10 Countries")

plt.tight_layout()

plt.savefig("top_countries.png")

plt.show()

# ======================================================
# Monthly Signup Trend
# ======================================================

month_order = [
'January','February','March','April','May','June',
'July','August','September','October','November','December'
]

monthly = (
df["signup_month"]
.value_counts()
.reindex(month_order)
)

plt.figure(figsize=(10,5))

monthly.plot(marker="o")

plt.title("Monthly Signup Trend")

plt.ylabel("Number of Students")

plt.grid(True)

plt.tight_layout()

plt.savefig("monthly_signup.png")

plt.show()

# ======================================================
# Signup Hour Distribution
# ======================================================

plt.figure(figsize=(10,5))

sns.histplot(
    df["engagement_intensity"],
    bins=30,
    kde=True
)

plt.title("Engagement Intensity Distribution")
plt.xlabel("Engagement Intensity")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("engagement_histogram.png")
plt.show()

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["engagement_intensity"]
)

plt.title("Boxplot of Engagement Intensity")
plt.xlabel("Engagement Intensity")

plt.tight_layout()

plt.savefig("engagement_boxplot.png")
plt.show()

plt.figure(figsize=(10,5))

sns.histplot(
    df["signup_hour"],
    bins=24,
    kde=True
)

plt.title("Signup Hour Distribution")

plt.tight_layout()

plt.savefig("signup_hour.png")

plt.show()

# ======================================================
# Opportunity Duration
# ======================================================

plt.figure(figsize=(10,5))

sns.histplot(
    df["opportunity_duration"],
    bins=30
)

plt.title("Opportunity Duration")

plt.tight_layout()

plt.savefig("duration_distribution.png")

plt.show()

# ======================================================
# Correlation
# ======================================================

numeric = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# ======================================================
# Outlier Detection
# ======================================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["opportunity_duration"]
)

plt.title("Opportunity Duration Outliers")

plt.tight_layout()

plt.savefig("outlier_boxplot.png")

plt.show()

# ======================================================
# Top Opportunity Categories
# ======================================================

print("\nTop Opportunity Categories\n")

print(df["opportunity_category"].value_counts())

# ======================================================
# Top Institutions
# ======================================================

print("\nTop Institutions\n")

print(df["institution_name"].value_counts().head(10))

# ======================================================
# Status Summary
# ======================================================

print("\nStatus Summary\n")

print(df["status_description"].value_counts())

# ======================================================
# Export Clean Dataset
# ======================================================

df.to_csv(
    "Week2_Cleaned_Dataset.csv",
    index=False
)

print("\nDataset exported successfully.")
