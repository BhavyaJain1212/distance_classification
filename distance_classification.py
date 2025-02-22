import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset (Iris dataset substitute)
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Label': np.random.choice([0, 1], 100)
}

df = pd.DataFrame(data)

# Splitting dataset
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Distance-Based Classification (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

