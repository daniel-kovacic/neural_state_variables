# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:22:29 2023

@author: kovac
"""
import tensorflow as tf

class DatasetUtil:
    """
    Class used for accessing video frames data in the form of
    tf.data.Dataset. Data needs to be stored correctly and a data-info json
    file is required for data to be accessable this way.

    Methods
    -------

    """

    def __init__(self, dataset_info, parts_hidden=False):
        """
        initializes a DatasetUtil object corresponding to a dataset.
        The object contains all relevant information about the data.

        Parameters
        ----------
        dataset_info : DatasetInfo
            Object which holds the most important info about .
        parts_hidden: boolean, optional
            Can be set to true if dataset_info contains a path to videoframes
            where part of the system are hidden.
            In this case the full frames are used for reconstruction the
            frames in which parts of the systems are hidden are used as input.
            The default is False.

        Raises
        ------
        Exception
            raises exception if number_of_points is no integer

        Returns
        -------
        None.

        """

        self.dataset_info = dataset_info
        self.parts_hidden = parts_hidden

        # number of data points is calcualted as a function of frames
        # used as input

    def __get_dataset_indices(self, mode):
        """
        Parameters
        ----------
        mode : str
            defines the mode one of ['train', 'val', 'test'].

        Raises
        ------
        Exception
            raises exception if  mode not in ['train', 'val', 'test'].

        Returns
        -------
        int list
            The directory indices used in the given mode.

        """
        if mode == "train":
            return self.dataset_info.train_ind
        elif mode == "val":
            return self.dataset_info.val_ind
        elif mode == "test":
            return self.dataset_info.test_ind
        elif mode == "all":
            return list(range(self.dataset_info.num_of_vids))
        raise Exception("mode needs to be one of  ['train', 'val', 'test', 'all']")

    @staticmethod
    def __random_index_generator(dataset_indices, data_tuples_per_folder):
        """
        generates random  dir, frame index tuple from given index list

        Parameters
        ----------
        dataset_indices : int list
            list of directory indices from which index should be generated.

        Yields
        ------
        dir_index :int
            random directory index.
        file_index : int
            random frame index.
        """
        while True:
            size = tf.shape(dataset_indices)[0]
            dir_index = tf.random.uniform(shape=(), maxval=size, dtype=tf.int32)
            frame_index = tf.random.uniform(
                shape=(), maxval=data_tuples_per_folder, dtype=tf.int32)
            yield dir_index, frame_index

    @staticmethod
    def __sequential_index_generator(dataset_indices, data_tuples_per_folder):
        """
        sequentialy generates dir, frame index tuple from given index list
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
            for frame_index in range(data_tuples_per_folder):
                yield dir_index, frame_index

    def get_dataset(self, index_mapper, data_tuples_per_vid, mode="train", sequential=False, batch_size=32):
        """
        creates and returns a tf.data.Dataset.

        Parameters
        ----------
        mode : str
            defines the mode one of ['training', 'validation', 'test']
        sequential : boolean, optional
            defines if returned dataset goes through data sequentially
            or randomly. The default is False.

        batch_size : int, optional
            The default is 32.
        index_mapper: function
            Function that maps indices of the form (video-index, first-frame-index) to the corresponding input-tensors.

        Returns
        -------
        dataset : tf.data.Dataset
            Dataset as specified by the arguments.

        """
        dataset_indices = self.__get_dataset_indices(mode)
        if sequential:
            def index_generator():
                return DatasetUtil.__sequential_index_generator(
                    dataset_indices, data_tuples_per_vid)
        else:
            def index_generator():
                return DatasetUtil.__random_index_generator(dataset_indices, data_tuples_per_vid)
        dataset = tf.data.Dataset.from_generator(
            index_generator, tf.uint32, output_shapes=2)
        dataset = dataset.map(index_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    pass