import pandas as pd

# Dataset load kar rahe hain
df = pd.read_csv("crop_production.csv")

print("Dataset Loaded Successfully ✅\n")

print("First 5 Rows:\n")
print(df.head())

print("\nColumns:\n", df.columns)

print("\nNull Values:\n")
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

# Label Encoding for categorical columns
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

print("\nAfter Encoding:\n")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features aur target define
X = df[["State", "Crop", "Season", "Area", "Cost"]]
y = df["Production"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model create
model = LinearRegression()

# Model train
model.fit(X_train, y_train)

print("\nModel Trained Successfully ✅")

# Prediction
predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("MSE:", mse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt
import seaborn as sns

# Production vs Area
plt.figure()
sns.scatterplot(x=df["Area"], y=df["Production"])
plt.title("Production vs Area")
plt.show()

# Production vs Cost
plt.figure()
sns.scatterplot(x=df["Cost"], y=df["Production"])
plt.title("Production vs Cost")
plt.show()
import joblib

joblib.dump(model, "agriculture_model.pkl")

print("\nModel Saved Successfully ✅")