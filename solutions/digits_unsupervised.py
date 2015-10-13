from sklearn.manifold import TSNE
from sklearn.decomposition import NMF

# Compute TSNE embedding
tsne = TSNE()
X_tsne = tsne.fit_transform(X)

# Visualize TSNE results
plt.title("All classes")
plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)

# build an NMF factorization of the digits dataset
nmf = NMF(n_components=16).fit(X)

# visualize the components
fig, axes = plt.subplots(4, 4)
for ax, component in zip(axes.ravel(), nmf.components_):
    ax.imshow(component.reshape(8, 8), cmap="gray", interpolation="nearest")
    ax.xticks(())
    ax.yticks(())
