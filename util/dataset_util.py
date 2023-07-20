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

    def get_dataset(self, index_mapper, data_tuples_per_vid, mode="train", sequential=False,
                    batch_size=32):
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
        steps_per_epoch : int
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
        autoencoder.summary()
        history = autoencoder.fit(training_dataset, epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_dataset,
                                  validation_steps=validation_steps,
                                  callbacks=[early_stopping, model_checkpoint])
        return history


    from dataset_info import DatasetInfo
    from index_mapper import *

    dataset_info = DatasetInfo.read_from_file("double_pendulum")
    dataset_util = DatasetUtil(dataset_info, dataset_info.get_data_tuples_per_vid(2))
    data=get_latent_memmap(dataset_info, "points", (dataset_info.get_data_tuples(), 2, 1, 1, 64))
    mapper = get_latent_array_mapper(
        data,
        dataset_info.frames_per_vid)
    dataset = dataset_util.get_dataset(mapper, dataset_info.get_data_tuples_per_vid(2))
    val_dataset = dataset_util.get_dataset(mapper, dataset_info.get_data_tuples_per_vid(2),
                                           mode="val")
    train_autoencoder(r"C:\Users\kovac\PycharmProjects\neural_state_variables\models\double_pendulum_latent_rec_3d_2frames",
                      dataset, val_dataset, 100)
