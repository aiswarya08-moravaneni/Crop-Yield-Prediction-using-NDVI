import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------
# 🔹 1. SEASONAL MODEL
# ---------------------------
seasonal_df = pd.read_csv("data/final_dataset.csv")

seasonal_df['Efficiency'] = seasonal_df['Yield'] / seasonal_df['NDVI']

seasonal_ml = pd.get_dummies(seasonal_df, columns=['District', 'Crop', 'Season'])

X_s = seasonal_ml.drop(['Year', 'Production', 'Yield', 'Efficiency'], axis=1)
y_s = seasonal_ml['Yield']

# Train-test split ✅
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

# Train model
seasonal_model = RandomForestRegressor(n_estimators=100, random_state=42)
seasonal_model.fit(X_train, y_train)

# Save model
pickle.dump(seasonal_model, open("models/seasonal_model.pkl", "wb"))
pickle.dump(X_s.columns, open("models/seasonal_columns.pkl", "wb"))

# Predict
y_pred = seasonal_model.predict(X_test)

# Save test results
test_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
test_df.to_csv("results/seasonal_test_results.csv", index=False)

# Save R2
r2 = r2_score(y_test, y_pred)
pickle.dump(r2, open("models/seasonal_r2.pkl", "wb"))

print(" Seasonal model + results saved")


# ---------------------------
# 🔹 2. YEARLY MODEL
# ---------------------------
yearly_df = pd.read_csv("data/final_total_dataset.csv")

yearly_df['Efficiency'] = yearly_df['Yield'] / yearly_df['NDVI']

yearly_ml = pd.get_dummies(yearly_df, columns=['District', 'Crop'])

X_y = yearly_ml.drop(['Year', 'Production', 'Yield', 'Efficiency'], axis=1)
y_y = yearly_ml['Yield']

# Train-test split ✅
X_train, X_test, y_train, y_test = train_test_split(X_y, y_y, test_size=0.2, random_state=42)

# Train model
yearly_model = RandomForestRegressor(n_estimators=100, random_state=42)
yearly_model.fit(X_train, y_train)

# Save model
pickle.dump(yearly_model, open("models/yearly_model.pkl", "wb"))
pickle.dump(X_y.columns, open("models/yearly_columns.pkl", "wb"))

# Predict
y_pred = yearly_model.predict(X_test)

# Save test results
test_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
test_df.to_csv("results/yearly_test_results.csv", index=False)

# Save R2
r2 = r2_score(y_test, y_pred)
pickle.dump(r2, open("models/yearly_r2.pkl", "wb"))

print(" Yearly model + results saved")