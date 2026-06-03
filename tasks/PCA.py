import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = sns.load_dataset('iris')

data = data[data['species'] != 'versicolor']

X = data.iloc[:, :4]
y = data['species']

#print(data, type(data))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)

setosa = y == 'setosa'
virginica = y == 'virginica'

plt.scatter(X_pca[setosa, 0], X_pca[setosa, 1], label='setosa')
plt.scatter(X_pca[virginica, 0], X_pca[virginica, 1], label='virginica')

plt.legend()
plt.show()