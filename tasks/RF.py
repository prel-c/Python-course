import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = sns.load_dataset('iris')

data = data[data['species'] != 'versicolor']

X = data.iloc[:, :2]
y = data['species'].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')

scatter = plt.scatter(
    X.iloc[:, 0], 
    X.iloc[:, 1], 
    c=y, 
    cmap='Set1', 
    edgecolor='k', 
    s=60
)

labels = data['species'].astype('category').cat.categories.tolist()

plt.legend(handles=scatter.legend_elements()[0], labels=labels, title="Species")
plt.title("Гиперплоскость Random Forest")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()