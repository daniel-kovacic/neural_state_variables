# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors


def reshape_array_for_levina_bickels_alg(array, dataset_info):
    return array.reshape((dataset_info.num_of_vids*dataset_info.get_data_tuples_per_vid(),
                          np.prod(dataset_info.get_latent_enc_shape())))


def levina_bickels_alg(points, neighbors_lower_bound=15,neighbors_upper_bound=25):
    """
    levina-Bickel's algorithm implementation.An average over different
    choices for the number of neighbors is used. The performance effect
    of the range for the number of neighbours is negligible.

    Parameters
    ----------
    points : numpy-array
        array with all points used for ID estimation
    neighbors_lower_bound : int, optional
        lower bound for number of neighbours used for estimation.
        The default is 15.
    neighbors_upper_bound : int, optional
        upper bound for number of neighbours used for estimation.
        The default is 25.

    Returns
    -------
    global_dim_appr: double
        calculated dimension of whole dataset
    aver_dimension : np.array
        calculated of intrinsic dimension estimations
    """

    # remove duplicates points
    points = np.unique([tuple(row) for row in points], axis=0)
    number_of_points = len(points)

    number_of_approximations = neighbors_upper_bound - neighbors_lower_bound

    nbrs = NearestNeighbors(n_neighbors=neighbors_upper_bound,
                            algorithm='auto').fit(points)

    distances, _ = nbrs.kneighbors(points)

    # implementation of the levina bickel's algorithm
    local_dimension_approximations = np.array(
        [(j - 2) / np.sum(np.log(distances[:, j:j + 1] / distances[:, 1:j]), axis=1)
         for j in range(neighbors_lower_bound, neighbors_upper_bound)])

    # average over results for range of approximations
    average_local_dimension_approximations = np.sum(
        local_dimension_approximations, axis=0) / number_of_approximations

    # average over local intrinsic dimension approximations
    global_dimension_approximations = np.sum(
        average_local_dimension_approximations, axis=0) / number_of_points

    return (global_dimension_approximations,
            average_local_dimension_approximations)
