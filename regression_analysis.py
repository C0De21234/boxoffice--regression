import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# 1. Generate synthetic data
# -------------------------
np.random.seed(1)
budget = np.random.randint(1000, 10000, 150)
box_office = budget * 3 * np.random.uniform(1.5, 2.5, 150)
rating = box_office / 1000 + np.random.normal(1, 5, 150)

df = pd.DataFrame({
    'Budget': budget,
    'BoxOffice': box_office,
    'Rating': rating
})

# -------------------------
# 2. Simple Linear Regression
# -------------------------
X_simple = df[['Budget']]
y_simple = df[['BoxOffice']]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=1
)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# Metrics - Simple Regression
r2_s = r2_score(y_test_s, y_pred_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)

# -------------------------
# 3. Multiple Linear Regression
# -------------------------
X_multi = df[['Budget', 'Rating']]
y_multi = df[['BoxOffice']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=1
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

# Metrics - Multiple Regression
r2_m = r2_score(y_test_m, y_pred_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)

# -------------------------
# 4. Visualization
# -------------------------
plt.figure(figsize=(12, 6))

# Simple Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(X_test_s.values.flatten(), y_test_s.values.flatten(),
            color='blue', label='Actual')
x_line = np.linspace(X_test_s.min(), X_test_s.max(), 100).reshape(-1, 1)
y_line = model_simple.predict(x_line)
plt.plot(x_line, y_line, color='red', label='Predicted')
plt.title(f'Simple Linear Regression\n(Budget → Box Office)\n'
          f'R²={r2_s:.2f}, MAE={mae_s:.0f}, MSE={mse_s:.0f}')
plt.xlabel('Budget')
plt.ylabel('Box Office')
plt.legend()
plt.grid(True)

# Multiple Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test_m, y_pred_m, color='green', label='Predicted vs Actual')
plt.plot([y_test_m.min(), y_test_m.max()],
         [y_test_m.min(), y_test_m.max()],
         'r--', lw=2, label='Perfect Fit Line')
plt.title(f"Multiple Linear Regression\n(Budget + Rating → Box Office)\n"
          f'R²={r2_m:.2f}, MAE={mae_m:.0f}, MSE={mse_m:.0f}')
plt.xlabel('Actual Box Office')
plt.ylabel('Predicted Box Office')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("plots.png", dpi=300)  # save figure for GitHub
plt.show()
