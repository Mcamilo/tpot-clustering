# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

clustering_config_dict = {

    # Clusterers
    'sklearn.cluster.AgglomerativeClustering': {
        'n_clusters': range(2, 23),
        'metric': ['euclidean'],
        'linkage': ['ward']
    },

    'sklearn.cluster.DBSCAN': {
        'eps': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'min_samples': [10, 25, 50],
        'metric': ['euclidean',],
        'leaf_size': [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    },

    'sklearn.cluster.KMeans': {
        'n_clusters': range(2, 23),
        'init': ['k-means++', 'random'],
    },

    'sklearn.cluster.MiniBatchKMeans': {
        'n_clusters': range(2, 23),
        'batch_size':[10, 25, 50]
    },

    'sklearn.cluster.SpectralClustering': {
        'n_clusters': range(2, 23),
        'eigen_solver': ['arpack', 'lobpcg', 'amg'],
        'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
    },

    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2']
    },

    'sklearn.preprocessing.StandardScaler': {
    },
    
    'sklearn.decomposition.PCA': {
        'n_components': [2, 3, 5, 10]
    },

    'sklearn.decomposition.FastICA': {
        'n_components': [2, 3, 5, 10]
    },
    
    # Selectors
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.1, 0.25]
    },

}
