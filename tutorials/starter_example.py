from tpotclustering import TPOTClustering
from sklearn.datasets import make_blobs

# Generate synthetic clustering data
X, y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.0, random_state=42)

scoring = "silhouette_score"

tpot_clustering = TPOTClustering(
    generations=5,
    population_size=10,
    verbosity=2,
    random_state=42,
    scoring=scoring
)
tpot_clustering.fit(X)
print("Optimized clustering score:", tpot_clustering.score(X))
tpot_clustering.export(f"tpot_clustering_{scoring}.py")
