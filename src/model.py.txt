import time

start = time.time()
from xgboost import XGBRegressor

model = XGBRegressor(
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",  # ✅ regression objective
    eval_metric="rmse"              # ✅ root mean square error
)


model.fit(X_train_split, y_train_split, verbose=True)
print("✅ Model trained in", round(time.time() - start, 2), "seconds")
