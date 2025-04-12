import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

file_path = "Electric_Vehicle_Population_Data (1).csv"
df = pd.read_csv(file_path)

df.dropna(subset=['Electric Vehicle Type', 'Make', 'Model', 'County', 'City', 'Model Year'], inplace=True)

print("=== HEAD ===")
print(df.head(), "\n")

print("=== TAIL ===")
print(df.tail(), "\n")

print("=== INFO ===")
print(df.info(), "\n")

print("=== DESCRIBE ===")
print(df.describe(include='all'), "\n")

print("=== SHAPE ===")
print("Rows:", df.shape[0], "Columns:", df.shape[1], "\n")

print("=== MISSING VALUES ===")
print(df.isnull().sum(), "\n")

type_counts = df['Electric Vehicle Type'].value_counts()
plt.figure(figsize=(6, 6))
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62'])
plt.title('Proportion of EV Types (BEV vs PHEV)')
plt.ylabel('')
plt.tight_layout()
plt.show()

type_year = df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack().fillna(0)
plt.figure(figsize=(10, 6))
type_year.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set3')
plt.title("Trend of EV Types Over Model Years")
plt.xlabel("Model Year")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.show()

top_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_makes.index, y=top_makes.values)
plt.title('Top 10 Electric Vehicle Manufacturers')
plt.xlabel('Make')
plt.ylabel('Registrations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df['Make_Model'] = df['Make'] + " " + df['Model']
top_models = df['Make_Model'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_models.index, y=top_models.values)
plt.title('Top 10 EV Models')
plt.xlabel('Make and Model')
plt.ylabel('Registrations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

top_counties = df['County'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_counties.index, y=top_counties.values)
plt.title('Top 10 Counties by EV Registration')
plt.xlabel('County')
plt.ylabel('Registrations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_cities = df['City'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cities.index, y=top_cities.values)
plt.title('Top 10 Cities by EV Registration')
plt.xlabel('City')
plt.ylabel('Registrations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

yearly = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly.index, y=yearly.values, marker='o', linewidth=2.5)
plt.title('EV Registrations Over Model Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Model Year', hue='Electric Vehicle Type', palette='Set2')
plt.title('Distribution of BEVs and PHEVs Per Year')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

avg_annual_miles = 11500
co2_per_mile_ice = 0.411
df['Estimated_CO2_Saved_kg'] = avg_annual_miles * co2_per_mile_ice
total_co2_saved = df['Estimated_CO2_Saved_kg'].sum() / 1e6

plt.figure(figsize=(7, 5))
sns.barplot(x=["Estimated Annual CO₂ Reduction (Million kg)"], y=[total_co2_saved], color="#8da0cb")
plt.title('Estimated Annual CO₂ Savings by EVs')
plt.ylabel('Million kg CO₂ Saved')
plt.tight_layout()
plt.show()

numerical_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 5))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numerical Features')
plt.tight_layout()
plt.show()

bev = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']['Model Year']
phev = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']['Model Year']
t_stat, p_value = stats.ttest_ind(bev, phev, equal_var=False)
print("T-Test Between Model Years of BEVs and PHEVs:")
print("T-statistic:", t_stat)
print("P-value:", p_value, "\n")

make_counts = df['Make'].value_counts()
z_scores = stats.zscore(make_counts)
outliers = make_counts[np.abs(z_scores) > 2]
print("Outlier Vehicle Makes Based on Z-Score Analysis:")
print(outliers, "\n")

shapiro_test = stats.shapiro(df['Model Year'].sample(n=500, random_state=1))
print("Shapiro-Wilk Test for Normality of Model Year:")
print("W-statistic:", shapiro_test[0])
print("P-value:", shapiro_test[1], "\n")