from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_valide)

mse = mean_squared_error(y_valide, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_valide, y_pred)
r2 = r2_score(y_valide, y_pred)
smape_score = smape(y_valide, y_pred)

print("SMAPE:", smape_score, "%")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)


download a csv file of predicted prices for test csv:

import pandas as pd
import numpy as np

# Load test CSV
TEST_CSV = "/kaggle/input/mlhack/test.csv"
test_df = pd.read_csv(TEST_CSV)

# Predict with both models
xgb_test_preds = xgb_model.predict(X_test_full)
mlp_test_preds = mlp_model(torch.tensor(X_test_full, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()

# Ensemble
final_test_preds = 0.6 * xgb_test_preds + 0.4 * mlp_test_preds

# Convert back from log
final_test_preds_exp = np.expm1(final_test_preds)

# Create submission DataFrame
submission_df = pd.DataFrame({
    "sample_id": test_df["sample_id"].iloc[:X_test_full.shape[0]],  # align sample_id with available rows
    "price": final_test_preds_exp.flatten()
})

# Save CSV
submission_df.to_csv("/kaggle/working/XGBoost_mlp_predictions.csv", index=False)
print("✅ Predictions saved to tabnet_test_predictions.csv")
