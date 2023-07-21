# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:26:19 2023

@author: kovac
"""

import os
import numpy as np
from dataset_util import DatasetUtil


def create_latent_space_array(dataset_info, number_of_frames=2, encoding_shape=(2, 1, 1, 64), filename="points",
                              memmap=True, is_neural=False):
    shape = (dataset_info.get_data_tuples(number_of_frames), *encoding_shape)

    if memmap:
        # initialize memmap of size equal to the number of data tuples
        latent_dir_path = dataset_info.get_latent_path()
        file_path = os.path.join(latent_dir_path, filename)
        if not os.path.exists(latent_dir_path):
            os.makedirs(latent_dir_path)
        return np.memmap(
            file_path,
            dtype='float32',
            mode='w+',
            shape=shape)
    else:
        return np.zeros(shape)


def create_encodings(dataset_info, dataset, autoencoder, number_of_frames=2, encoding_shape=(2, 1, 1, 64),
                     filename="points", memmap=True, is_neural=False):
    """
    uses DatasetUtil instance and provided autoencoder to store
    latent space encodings as np.memmap

    Parameters
    ----------
    dataset_util : DatasetUtil
        dataset util giving access and information about dataset which is to
        be encoded by the provided autoencoder
    autoencoder : tf.keras.model
        dynamics prediction autoencoder used for encoding video frames from
        dataset
    dim : int, optional
        2/3 depending on if the autoencoder uses 2D or 3D convoltion.
        The default is 3.
    encoding_shape : tuple, optional
        latent space dimension of the used autoencoder.
        The default is (2,1,1,64).
    filename : str, optional
        name of the encoded np.memmap file. The default is "points".

    Returns
    -------
    np.array
        array with the latent encodings

    """

    empty_data_array = create_latent_space_array(dataset_info, number_of_frames=number_of_frames,
                                                 encoding_shape=encoding_shape, filename=filename,
                                                 memmap=memmap, is_neural=is_neural)

    # extract encoder part of autoencoder
    encoder = autoencoder.layers[0]

    data_tuples_per_vid = dataset_info.get_data_tuples_per_vid(number_of_frames)
    # create all predictions folder by folder and append to np.memmap
    for i, data in enumerate(dataset):
        predictions = encoder.predict(data[0])
        empty_data_array[i * data_tuples_per_vid:(i + 1) * data_tuples_per_vid] = predictions[:]


if __name__ == "__main__":
    from dataset_info import DatasetInfo
    from dataset_util import DatasetUtil
    import tensorflow as tf
    from index_mapper import get_data_preprocessor
    dataset_info = DatasetInfo.read_from_file("double_pendulum")
    dataset_util = DatasetUtil(dataset_info)
    dataset = dataset_util.get_dataset(index_mapper=get_data_preprocessor(dataset_info), mode="all",data_tuples_per_vid=dataset_info.get_data_tuples_per_vid(2),
                                       batch_size=dataset_info.get_data_tuples_per_vid(2), sequential=True)
    autoencoder = tf.keras.models.load_model(
        r"C:\Users\kovac\PycharmProjects\neural_state_variables\models\double_pendulum_dyn_pred_3d_2frames")
    
    create_latent_space_predictions(dataset_info, dataset, autoencoder, encoding_shape=(2, 1, 1, 64))
