# -*- coding: utf-8 -*-

"""This file is part of the TPOT-Clustering library.

TPOT-Clustering is a fork of the TPOT library, extended to support unsupervised machine learning (clustering) tasks.

The original TPOT library was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT-Clustering is maintained by:
    - Matheus Camilo da Silva (matheus.camilo@phd.units.it)
    - Sylvio Barbon Junior (sylvio.barbon@units.it)
    - with additional contributions from the open source community

TPOT-Clustering is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT-Clustering is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program. If not, see <http://www.gnu.org/licenses/>.

Original TPOT project: https://github.com/EpistasisLab/tpot
TPOT-Clustering project: https://github.com/Mcamilo/tpot-clustering
"""


import numpy as np
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.metrics.cluster._unsupervised import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity

# class UnsupervisedScorer:
#     def __init__(self, metric, greater_is_better=True) -> None:
#         self.metric = metric
#         self.greater_is_better = greater_is_better
#     def __call__(self, estimator, X):
#         try:
#             cluster_labels = estimator.fit_predict(X)
#             if self.greater_is_better:
#                 return self.metric(X, cluster_labels) if len(set(cluster_labels)) > 1 else -float('inf') 
#             return -self.metric(X, cluster_labels) if len(set(cluster_labels)) > 1 else -float('inf') 
#         except Exception as e:
#             raise TypeError(f"{self.metric.__name__} is not a valid unsupervised metric function")
class UnsupervisedScorer:
    def __init__(self, metric, greater_is_better=True, use_transformed_X=True):
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.use_transformed_X = use_transformed_X

    def __call__(self, estimator, X):
        try:
            # Get labels from fitted estimator
            cluster_labels = estimator.predict(X)

            # Get transformed X if applicable
            if self.use_transformed_X and hasattr(estimator, "named_steps"):
                for name, step in list(estimator.named_steps.items())[:-1]:  # exclude final estimator
                    X = step.transform(X)

            score = self.metric(X, cluster_labels) if len(set(cluster_labels)) > 1 else -float('inf')
            return score if self.greater_is_better else -score
        except Exception as e:
            raise TypeError(f"{self.metric.__name__} failed: {e}")

class IntraClusterCosineSimilarity(UnsupervisedScorer):
    def __init__(self):
        # Higher is better — more semantically coherent clusters
        super().__init__(self.metric, greater_is_better=True)

    def metric(self, X, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        scores = []
        for label in unique_labels:
            cluster_points = X[cluster_labels == label]
            if len(cluster_points) < 2:
                continue  # Skip singleton clusters
            sim_matrix = cosine_similarity(cluster_points)
            upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
            avg_sim = np.mean(sim_matrix[upper_tri_indices])
            scores.append(avg_sim)
        return np.mean(scores) if scores else -1.0  # Return low score if no valid clusters

class CombinedInternalScore(UnsupervisedScorer):
    def __init__(self, weights=None):
        """
        weights: dict with keys 'cosine', 'ch', 'db' — you can customize importance
        """
        super().__init__(self.metric, greater_is_better=True)
        self.weights = weights or {
            'cosine': 0.8,
            'ch': 0.2,
            # 'db': 0.2  # note: inverted since lower is better
        }

    def metric(self, X, cluster_labels):
        if len(set(cluster_labels)) <= 1:
            return -float('inf')  # not a valid clustering

        try:
            # 1. Intra-cluster cosine similarity
            cosine_sim = IntraClusterCosineSimilarity().metric(X, cluster_labels)

            # 2. Calinski-Harabasz
            ch_score = calinski_harabasz_score(X, cluster_labels)

            # 3. Davies-Bouldin (lower is better → invert)
            # db_score = davies_bouldin_score(X, cluster_labels)
            # db_score = 1 / (1 + db_score)

            # Normalize each (optional: if scale imbalance is an issue)
            # Combine using weights
            combined = (
                self.weights['cosine'] * cosine_sim +
                self.weights['ch'] * ch_score
                # self.weights['ch'] * ch_score +
                # self.weights['db'] * db_score
            )
            return combined
        except Exception as e:
            return -float('inf')


SCORERS = {name: get_scorer(name) for name in get_scorer_names()}

SCORERS['silhouette_score'] = UnsupervisedScorer(silhouette_score)
SCORERS['davies_bouldin_score'] = UnsupervisedScorer(davies_bouldin_score, greater_is_better=False)
SCORERS['calinski_harabasz_score'] = UnsupervisedScorer(calinski_harabasz_score)
SCORERS['silhouette_samples'] = UnsupervisedScorer(silhouette_samples)
SCORERS['intra_cluster_cosine'] = IntraClusterCosineSimilarity()
SCORERS['combined_internal_score'] = CombinedInternalScore()


