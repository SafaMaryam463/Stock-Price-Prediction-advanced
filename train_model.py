import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("stocks/AAPL.csv")

# Keep only Close
df = df[['Close']]

# Create prediction column
df['Prediction'] = df['Close'].shift(-30)
df.dropna(inplace=True)

# Split data
X = df[['Close']]
y = df['Prediction']

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully!")
