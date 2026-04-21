import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


sns.set_theme(style="whitegrid", palette="deep")


# LOAD DATA

df = pd.read_csv(r"C:\Users\sharm\Downloads\fake_job_dataset_10000.csv")

print(df.head())
print(df.info())


# DATA CLEANING

df['Remote'] = df['Remote'].map({'Yes':1, 'No':0})
df['Has Company Logo'] = df['Has Company Logo'].map({'Yes':1, 'No':0})
df['Fraudulent'] = df['Fraudulent'].map({'Yes':1, 'No':0})

df.fillna(0, inplace=True)

print("\nMissing Values:\n", df.isnull().sum())


#  VISUALIZATIONS


# 1. Count Plot
plt.figure()
sns.countplot(x='Fraudulent', hue='Fraudulent', data=df, palette='Set2', legend=False)
plt.title("Real vs Fake Job Postings", weight='bold')
plt.tight_layout()
plt.show()

# 2. Pie Chart
plt.figure()
counts = df['Fraudulent'].value_counts()
plt.pie(counts, labels=['Real','Fake'], autopct='%1.1f%%',
        colors=sns.color_palette("pastel"), wedgeprops={'edgecolor':'black'})
plt.title("Fake vs Real Jobs Distribution", weight='bold')
plt.tight_layout()
plt.show()

# 3. Histogram
plt.figure()
sns.histplot(df['Fraud Score'], kde=True, bins=20, color='#4C72B0')
plt.title("Fraud Score Distribution", weight='bold')
plt.tight_layout()
plt.show()

# 4. Remote vs Fraud
plt.figure()
sns.countplot(x='Remote', hue='Fraudulent', data=df, palette='coolwarm')
plt.title("Remote Jobs vs Fraud", weight='bold')
plt.tight_layout()
plt.show()

# 5. Experience vs Fraud
plt.figure()
sns.countplot(x='Required Experience', hue='Fraudulent', data=df, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Experience vs Fraud", weight='bold')
plt.tight_layout()
plt.show()

# 6. Company Logo vs Fraud
plt.figure()
sns.countplot(x='Has Company Logo', hue='Fraudulent', data=df, palette='coolwarm')
plt.title("Company Logo vs Fraud", weight='bold')
plt.tight_layout()
plt.show()

# 7. Top Fake Job Titles
fake_jobs = df[df['Fraudulent'] == 1]
plt.figure()
fake_jobs['Job Title'].value_counts().head(10).plot(
    kind='bar', color='#55A868', edgecolor='black'
)
plt.title("Top Fake Job Titles", weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Boxplot (Description Length) - FIXED
plt.figure()
sns.boxplot(x='Fraudulent', y='Description Length',
            hue='Fraudulent', data=df,
            palette='coolwarm', legend=False)
plt.title("Description Length vs Fraud", weight='bold')
plt.tight_layout()
plt.show()

# 9. Boxplot (Fraud Score) - FIXED
plt.figure()
sns.boxplot(x='Fraudulent', y='Fraud Score',
            hue='Fraudulent', data=df,
            palette='coolwarm', legend=False)
plt.title("Fraud Score vs Fraud", weight='bold')
plt.tight_layout()
plt.show()

# 10. Donut Chart
plt.figure()
counts = df['Fraudulent'].value_counts()
plt.pie(counts, autopct='%1.1f%%',
        colors=['#66c2a5','#fc8d62'],
        wedgeprops={'edgecolor':'white'})
centre_circle = plt.Circle((0,0),0.60,fc='white')
plt.gca().add_artist(centre_circle)
plt.title("Fake vs Real Jobs", weight='bold')
plt.tight_layout()
plt.show()


#  data training

X = df[['Fraud Score', 'Description Length', 'Remote', 'Has Company Logo']]
y = df['Fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# EVALUATION

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# CORRELATION MATRIX

plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, linecolor='white')
plt.title("Correlation Matrix", weight='bold')
plt.tight_layout()
plt.show()


#  SCATTER PLOT

plt.figure()
sns.scatterplot(x='Fraud Score', y='Description Length',
                hue='Fraudulent', data=df,
                palette='Set1', alpha=0.7)
plt.title("Fraud Score vs Description Length", weight='bold')
plt.tight_layout()
plt.show()
