# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:54:45 2023

@author: kovac
"""
import tensorflow as tf


def train_autoencoder(autoencoder, training_dataset, validation_dataset,
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
    callbacks = []
    # stopp the model if it does not improve for a number of epochs and restore best weights
    if patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True))

    # saves the best model
    if save_path:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path, monitor='val_loss', save_best_only=True))

    # train model using stable learning rate, Adam optimizer, MAE for clear edges
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
    history = autoencoder.fit(training_dataset, epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=validation_dataset,
                              validation_steps=validation_steps,
                              callbacks=callbacks)
    return history


if __name__ == "__main__":
    """
    dataset_util = LatentDatasetUtil(DatasetUtil("harmonic_2body"))
    train_dataset = dataset_util.get_dataset()
    val_dataset = dataset_util.get_dataset(mode="validation")
    train_autoencoder(ModelUtil.get_model_path("harmonic_2body", is_latent=True), 
                      train_dataset, val_dataset,
                      1700, epochs=2000, patience= 1000, validation_steps=100)
    """
