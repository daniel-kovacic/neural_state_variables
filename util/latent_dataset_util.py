# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:43:09 2023

@author: kovac
"""

import tensorflow as tf
import numpy as np


def _random_index_generator(dataset_indices, data_tuples_per_folder):
    """
    generates random indices corresponding to original training/validation/
    test data

    Parameters
    ----------
    dataset_indices : int list
        DESCRIPTION.
    data_tuples_per_folder : int
        data tuples per folder in original dataset.

    Yields
    ------
    TYPE
        DESCRIPTION.

    """
    while True:
        size = tf.shape(dataset_indices)[0]
        dir_index = tf.random.uniform(shape=(), maxval=size, dtype=tf.int32)

        # chooses latent space data originating from folder in correct category
        yield (dataset_indices[dir_index] * data_tuples_per_folder +
               tf.random.uniform(
                   shape=(), maxval=data_tuples_per_folder, dtype=tf.int32))


def _sequential_index_generator(dataset_indices,
                                data_tuples_per_folder):
    """
    sequentially generates dir, frame index tuple from given index list
    Parameters
    ----------
    dataset_indices : int list
        list of directory indices from which index should be generated.

    Yields
    ------
    dir_index :int
        sequential directory index.
    file_index : int
        sequential frame index.
    """

    for dir_index in dataset_indices:
        for file_index in range(data_tuples_per_folder):
            yield dir_index * data_tuples_per_folder + file_index


class LatentDatasetUtil:
    """
    Class used for accessing encoded data from dynamics-prediction-autoencoder
    stored in np.memmap files.
    Used to train latent reconstruction autoencoders.

    Methods
    -------

    """

    def __init__(self, dataset_util, filename="points", latent_space_shape=(2, 1, 1, 64)):
        """

        Parameters
        ----------
        dataset_util : DatasetUtil
            DatasetUtil of the original video frames used to extract relevant
            information about the latent space data.
        filename : str, optional
            name of the file where the latent data is stored.
            The default is "points".
        latent_space_shape : float-tuple, optional
            Dimensions of the latent space. The default is (2,2,2,64).

        Returns
        -------
        None.

        """

        dir_path = "../latent_data/"
        self.dataset_util = dataset_util
        self.path = dir_path + dataset_util.data_title + "/" + filename
        self.number_of_points = dataset_util.number_of_data_tuples
        self.latent_space_shape = latent_space_shape

        # reuse training/validation/test split of original dataset
        self.training_dataset_indices = dataset_util.training_dataset_indices
        self.validation_dataset_indices = dataset_util.validation_dataset_indices
        self.test_dataset_indices = dataset_util.test_dataset_indices

        self.data = np.memmap(
            self.path, dtype='float32', mode='r',
            shape=(self.number_of_points, *self.latent_space_shape))

    def _get_dataset_indices(self, mode):
        """


        Parameters
        ----------
        mode : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            raises exception if mode is not in ['training', 'validation', 'test'].

        Returns
        -------
        int numpy-array
            list of indices of original dataset corresponding to the chosen
            mode.

        """
        if mode == "training":
            return self.training_dataset_indices
        elif mode == "validation":
            return self.validation_dataset_indices
        elif mode == "test":
            return self.test_dataset_indices
        elif mode == "all":
            return list(range(self.dataset_util.number_of_folders))
        raise Exception("mode needs to be one of  ['training', 'validation', 'test']")

    def load_data(self, i):
        """
        loads data from specified index
        Parameters
        ----------
        i : int
            index of np.memmap array from which data should be loaded.

        Returns
        -------
        x : numpy-array
            latent space data.
        x : numpy-array
            latent space data.

        """
        x = self.data[i]
        return x, x

    def get_dataset(self, mode="training", batch_size=32, sequential=False):
        """
        creates dataset using minimal pipeline
        Parameters
        ----------
        mode : str, optional
            has to be one of: ['training', 'validation', 'test'].
            The default is "training".
        batch_size : int, optional
            batch size of returned dataset. The default is 32.
        sequential : boolean, optional
            defines if the dataset should be traversed sequentially or in random order.
        Returns
        -------
        dataset : tf.data.Dataset
            latent space dataset used for training
        """
        dataset_indices = self._get_dataset_indices(mode)
        if sequential:
            def index_generator():
                return _sequential_index_generator(
                    dataset_indices, self.dataset_util.data_tuples_per_folder)
        else:
            def index_generator():
                return _random_index_generator(
                    dataset_indices, self.dataset_util.data_tuples_per_folder)

        dataset = tf.data.Dataset.from_generator(
            index_generator, tf.uint32, output_shapes=())
        dataset = dataset.map(
            lambda i: tf.py_function(
                func=self.load_data,
                inp=[i],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
