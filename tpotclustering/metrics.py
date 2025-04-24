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

class UnsupervisedScorer:
    def __init__(self, metric, greater_is_better=True) -> None:
        self.metric = metric
        self.greater_is_better = greater_is_better
    def __call__(self, estimator, X):
        try:
            cluster_labels = estimator.fit_predict(X)
            if self.greater_is_better:
                return self.metric(X, cluster_labels) if len(set(cluster_labels)) > 1 else -float('inf') 
            return -self.metric(X, cluster_labels) if len(set(cluster_labels)) > 1 else -float('inf') 
        except Exception as e:
            raise TypeError(f"{self.metric.__name__} is not a valid unsupervised metric function")
        
SCORERS = {name: get_scorer(name) for name in get_scorer_names()}

SCORERS['silhouette_score'] = UnsupervisedScorer(silhouette_score)
SCORERS['davies_bouldin_score'] = UnsupervisedScorer(davies_bouldin_score, greater_is_better=False)
SCORERS['calinski_harabasz_score'] = UnsupervisedScorer(calinski_harabasz_score)
SCORERS['silhouette_samples'] = UnsupervisedScorer(silhouette_samples)

