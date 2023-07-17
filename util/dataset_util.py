# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:22:29 2023

@author: kovac
"""
import os
import tensorflow as tf
from index_mapper import *


class DatasetUtil:
    """
    Class used for accessing video frames data in the form of
    tf.data.Dataset. Data needs to be stored correctly and a data-info json
    file is required for data to be accessable this way.

    Methods
    -------

    """

    def __init__(
            self, dataset_info, number_of_frames=2, number_of_points=None,
            parts_hidden=False):
        """
        initializes a DatasetUtil object corresponding to a dataset.
        The object contains all relevant information about the data.

        Parameters
        ----------
        dataset_info : DatasetInfo
            Object which holds the most important info about .
        number_of_frames : int, optional
            number of frames that the corresponding tf.data.Dataset should
            provide for a single prediction.
            The default is 2.
        number_of_points : int, optional
            used for creating learning curves.
            number of data points used from the original dataset.
            If it is not specified all the original data is used.
            The default is None.
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
        self.number_of_frames = number_of_frames
        self.data_tuples_per_folder = dataset_info.frames_per_vid - self.number_of_frames
        self.number_of_data_tuples = (
                self.data_tuples_per_folder * dataset_info.num_of_vids)

        # if number_of_points != None: The used dataset is reduced
        # to the required size
        if number_of_points is not None:
            if not isinstance(number_of_points, int):
                raise Exception("number_of_points needs to be an integer or None")
            self.dataset_info.frames_per_vid = (
                    number_of_points // len(dataset_info.train_ind)
                    + number_of_frames)
            number_of_training_folders = (number_of_points // (dataset_info.frames_per_vid - number_of_frames + 1))
            self.dataset_info.train_ind = dataset_info.train_ind[:number_of_training_folders]
            self.number_of_data_tuples = number_of_points

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
        equentialy generates dir, frame index tuple from given index list
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

    def get_dataset(self, index_mapper, mode="train", sequential=False, dim=3, batch_size=32):
        """
        creates and returns a tf.data.Dataset.

        Parameters
        ----------
        mode : str
            defines the mode one of ['training', 'validation', 'test']
        sequential : boolean, optional
            defines if returned dataset goes through data sequentially
            or randomly. The default is False.
        dim : int, optional
            2/3 defines if returned dataset provides data
            for 2D or 3D convolution. The default is 3.

        batch_size : int, optional
            The default is 32.

        Returns
        -------
        dataset : tf.data.Dataset
            Dataset as specified by the arguments.

        """
        dataset_indices = self.__get_dataset_indices(mode)
        if sequential:
            def index_generator():
                return DatasetUtil.__sequential_index_generator(
                    dataset_indices, self.data_tuples_per_folder)
        else:
            def index_generator():
                return DatasetUtil.__random_index_generator(dataset_indices, self.data_tuples_per_folder)
        dataset = tf.data.Dataset.from_generator(
            index_generator, tf.uint32, output_shapes=2)

        dataset = dataset.map(index_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_preprocessed_dir_frames(self, directory_index, dim=3):
        """
        preprocesses and returns all frames from directory with given index

        Parameters
        ----------
        directory_index : int
            directory index.
        dim : int, optional
            Either 2 or 3. An extra dimension is added for 3D convolution.
            The default is 3.

        Returns
        -------
        list
            DESCRIPTION.

        """
        dir_path = os.path.join(self.path, str(directory_index))
        image_paths = [os.path.join(dir_path, str(i) + ".png") for i
                       in range(self.files_per_folder)]
        if dim == 2:
            return [image
                    for image in self._get_normalized_images(image_paths)]
        return [tf.expand_dims(image, axis=0)
                for image in self._get_normalized_images(image_paths)]


if __name__ == "__main__":
    def train_autoencoder(autoencoder_path, training_dataset, validation_dataset,
                          steps_per_epoch, save_path=None, patience=50, epochs=300, learning_rate=5e-4,
                          validation_steps=100, loss='MSE'):
        """
        function for training autoencoders, which defines some defaults for
        hyperparameters

        Parameters
        ----------
        autoencoder_path : str
            path to the autoencoder which should be trained.
        training_dataset : tf.data.Dataset
            dataset used for training.
        validation_dataset : tf.data.Dataset
            dataset used for validation during training.
        steps_per_epoch : tf.data.Dataset
            dataset used for training.
        save_path : str, optional
            path where the trained model should be saved
            If no path is given it's saved to autoencoder_path.
            The default is None.
        patience : int, optional
            defines number of epochs with no improvement after which
            training is stopped.
            The default is 50.
        epochs : int, optional
            maximal number of epochs during training. The default is 300.
        learning_rate : float, optional
            used learning rate. The default is 5e-4.
        validation_steps : int, optional
            pred-steps used for validation after each epoch. The default is 100.

        Returns
        -------
        None.

        """
        autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)

        # stopp the model if it does not improve for 50 epochs and restore best weights
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience, restore_best_weights=True)

        # saves the best model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path if save_path else autoencoder_path,
            monitor='val_loss', save_best_only=True)

        # train model using stable learning rate, Adam optimizer, MAE for clear edges
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
        history = autoencoder.fit(training_dataset, epochs=epochs,
                                  steps_per_epoch=steps_per_epoch
                                  , validation_data=validation_dataset,
                                  validation_steps=validation_steps,
                                  callbacks=[early_stopping, model_checkpoint])
        return history
    from dataset_info import DatasetInfo

    dataset_util = DatasetUtil(DatasetInfo.read_from_file("double_pendulum"))
    dataset1 = dataset_util.get_dataset(dim=2)
    dataset2 = dataset_util.get_dataset(mode="val", dim=2)
    next(iter(dataset2))
    train_autoencoder("../models/untrained_dynamics_pred_2d", dataset1, dataset2, 100)
