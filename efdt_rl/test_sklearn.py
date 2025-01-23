from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Model initialization and fitting
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Output
print("Predictions:", predictions)
