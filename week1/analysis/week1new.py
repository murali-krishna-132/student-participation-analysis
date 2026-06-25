import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Make sure the CSV file is in the same folder as this Python script
file_path = 'RIT+Opportunity+Wise+Data+-+Sheet1.csv'

print("Loading dataset...")
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")
except FileNotFoundError:
    print(f"Error: Could not find the file '{file_path}'. Please ensure it is in the same directory.")
    exit()

# ==========================================
# STEP 2: Data Cleaning (Week 1 Prep)
# ==========================================
print("Cleaning the data...")

# 1. Drop missing rows in specific columns (only 5 missing)
df_clean = df.dropna(subset=['Institution Name', 'Current/Intended Major']).copy()

# 2. Fix corrupted timestamps in 'Apply Date' (extract just the date part)
df_clean['Apply Date'] = df_clean['Apply Date'].apply(lambda x: str(x).split(' ')[0] if pd.notnull(x) else x)

# 3. Fill missing 'Opportunity Start Date' with a placeholder
df_clean['Opportunity Start Date'] = df_clean['Opportunity Start Date'].fillna('Not Specified')

# 4. Convert 'Date of Birth' to proper datetime format
df_clean['Date of Birth'] = pd.to_datetime(df_clean['Date of Birth'], errors='coerce')

# 5. Calculate Age for analysis purposes (assuming current year is 2024 for context)
current_year = datetime.now().year
df_clean['Age'] = current_year - df_clean['Date of Birth'].dt.year

# Save the cleaned dataset to a new file
clean_file_path = 'Cleaned_RIT_Opportunity_Data.csv'
df_clean.to_csv(clean_file_path, index=False)
print(f"Data cleaning complete. Cleaned data saved to '{clean_file_path}'.\n")


# ==========================================
# STEP 3: Summary Statistics
# ==========================================
print("--- SUMMARY STATISTICS ---")

print("\n1. Participation by Opportunity Category:")
print(df_clean['Opportunity Category'].value_counts())

print("\n2. Applicant Status Outcomes:")
print(df_clean['Status Description'].value_counts())

print("\n3. Gender Distribution:")
print(df_clean['Gender'].value_counts())


# ==========================================
# STEP 4: Exploratory Data Analysis (Visualizations)
# ==========================================
print("\nGenerating visualizations... (Close each window to view the next chart)")

# Set the visualization style
sns.set_theme(style="whitegrid")

# Figure 1: Distribution of Opportunity Categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='Opportunity Category', order=df_clean['Opportunity Category'].value_counts().index, palette='viridis')
plt.title('Distribution of Opportunities by Category', fontsize=14)
plt.ylabel('Number of Participants')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 2: Application Outcomes (Status Description)
plt.figure(figsize=(12, 6))
sns.countplot(data=df_clean, y='Status Description', order=df_clean['Status Description'].value_counts().index, palette='magma')
plt.title('Overall Applicant Status Outcomes', fontsize=14)
plt.xlabel('Count')
plt.ylabel('Status')
plt.tight_layout()
plt.show()

# Figure 3: Gender Distribution across Opportunity Categories
plt.figure(figsize=(12, 6))
# Filter out sparse genders for a cleaner chart
df_gender = df_clean[df_clean['Gender'].isin(['Male', 'Female'])]
sns.countplot(data=df_gender, x='Opportunity Category', hue='Gender', palette='Set2')
plt.title('Gender Distribution by Opportunity Category', fontsize=14)
plt.ylabel('Count')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

# Figure 4: Age Distribution of Participants
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Age'].dropna(), bins=20, kde=True, color='teal')
plt.title('Age Distribution of Participants', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")