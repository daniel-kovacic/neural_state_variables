# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:04:59 2023

"""
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, BatchNormalization, Dense,
                                    Flatten, Reshape, Dropout, Activation)
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import numpy as np


def get_3d_conv_layer(filters, kernel_size=(2, 3, 3), strides=(1, 2, 2),
                      activation='relu', padding='same', **kwargs):
    """
    creates a 3d convolutional layer with default parameters as needed for
    the dynamics prediction autoencoder

    Parameters
    ----------
    filters : int
        number of filters.
    kernel_size : int tuple, optional
        The default is (2,3,3).
    strides : int tuple, optional
        The default is (1,2,2).
    activation : str/tf.keras.activations, optional
        activation function used for the layer. The default is 'relu'.
    padding : str, optional
        The default is 'same'.
    **kwargs : dict
        other kwargs that should be passed to the constructed layer.

    Returns
    -------
    tf.keras.layers.Conv3D
    """

    return Conv3D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  activation=activation,
                  padding=padding,
                  **kwargs)


def get_3d_conv_transpose_layer(filters, kernel_size=(2, 3, 3), strides=(1, 2, 2),
                                activation='relu', padding='same', **kwargs):
    """
    creates a 3d-convolutional-transpose layer with default parameters
    as needed for the dynamics prediction autoencoder

    Parameters
    ----------
    filters : int
        number of filters.
    kernel_size : int tuple, optional
        The default is (2,3,3).
    strides : int tuple, optional
        The default is (1,2,2).
    activation : str/tf.keras.activations, optional
        activation function used for the layer. The default is 'relu'.
    padding : str, optional
        The default is 'same'.
    **kwargs : dict
        other kwargs that should be passed to the constructed layer.

    Returns
    -------
    tf.keras.layers.Conv3DTranspose

    """
    return Conv3DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           activation=activation,
                           padding=padding,
                           **kwargs)


def get_2d_conv_layer(filters, kernel_size=(4, 4), strides=(2, 2),
                      activation='relu', padding='same', **kwargs):
    """
    creates a 2d-convolutional layer with default parameters
    as needed for the dynamics prediction autoencoder

    Parameters
    ----------
    filters : int
        number of filters.
    kernel_size : int tuple, optional
        The default is (2,3,3).
    strides : int tuple, optional
        The default is (1,2,2).
    activation : str/tf.keras.activations, optional
        activation function used for the layer. The default is 'relu'.
    padding : str, optional
        The default is 'same'.
    **kwargs : dict
        other kwargs that should be passed to the constructed layer.

    Returns
    -------
    tf.keras.layers.Conv2D

    """
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  activation=activation,
                  padding=padding,
                  **kwargs)


def get_2d_conv_transpose_layer(filters, kernel_size=(4, 4), strides=(2, 2),
                                activation='relu', padding='same', **kwargs):
    """
    creates a 2d-convolutional-transpose layer with default parameters
    as needed for the dynamics prediction autoencoder

    Parameters
    ----------
    filters : int
        number of filters.
    kernel_size : int tuple, optional
        The default is (4,4).
    strides : int tuple, optional
        The default is (2,2).
    activation : str/tf.keras.activations, optional
        activation function used for the layer. The default is 'relu'.
    padding : str, optional
        The default is 'same'.
    **kwargs : dict
        other kwargs that should be passed to the constructed layer.

    Returns
    -------
    tf.keras.layers.Conv2D

    """
    return Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           activation=activation,
                           padding=padding,
                           **kwargs)


def dynamics_autoencoder_def_3d(input_shape=(None, 128, 128, 3)):
    """
    Definition for 3d dynamics prediction autoencoder. Similar to the
    2D version.
    Latent space reduced by a factor of 4 compared to original space.
    Unnecessarily high dimensional

    Parameters
    ----------
    input_shape : int tuple, optional
        fixes input shape so that summary function of model gives more information.
        The default is (None,None, None, 3).

    Returns
    -------
    conv_autoencoder : tf.keras.Model
        untrained dynamics prediction autoencoder.

    """

    # Encoder
    conv_encoder = Sequential([
        get_3d_conv_layer(32, input_shape=input_shape),
        BatchNormalization(),
        get_3d_conv_layer(32),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(128),
        BatchNormalization(),
        get_3d_conv_layer(128, activation='sigmoid')])

    # Decoder
    conv_decoder = Sequential([
        get_3d_conv_transpose_layer(128),
        BatchNormalization(),
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(3, activation="sigmoid")])

    conv_autoencoder = Sequential([conv_encoder, conv_decoder])

    return conv_autoencoder


def dynamics_autoencoder_def_3d_updated(input_shape=(None, 128, 128, 3)):
    """
    Definition for 3d dynamics prediction autoencoder.
    Latent space reduced by a factor of 192 compared to original space.
    Capable of providing stable video predictions

    Parameters
    ----------
    input_shape : int tuple, optional
        fixes input shape so that summary function of model gives more information.
        The default is (None,None, None, 3).

    Returns
    -------
    conv_autoencoder : tf.keras.Model
        untrained dynamics prediction autoencoder.

    """

    # Encoder
    conv_encoder = Sequential([
        get_3d_conv_layer(32, input_shape=input_shape),
        BatchNormalization(),
        get_3d_conv_layer(32),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64, activation="sigmoid")])

    # Decoder
    conv_decoder = Sequential([
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(3, activation="sigmoid")])

    conv_autoencoder = Sequential([conv_encoder, conv_decoder])

    return conv_autoencoder


def dynamics_autoencoder_def_3d_updated2(input_shape=(None, 128, 128, 3)):
    """
    Definition for 3d dynamics prediction autoencoder.
    Latent space reduced by a factor of 384 compared to original space.
    Capable of providing stable video predictions
    Architecture used for Models in Bachelor Thesis

    Parameters
    ----------
    input_shape : int tuple, optional
        fixes input shape so that summary function of model gives more information.
        The default is (None,None, None, 3).

    Returns
    -------
    conv_autoencoder : tf.keras.Model
        untrained dynamics prediction autoencoder.

    """

    # Encoder
    conv_encoder = Sequential([
        get_3d_conv_layer(32, input_shape=input_shape),
        BatchNormalization(),
        get_3d_conv_layer(32),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64),
        BatchNormalization(),
        get_3d_conv_layer(64, strides=(1, 4, 4), activation="sigmoid")])

    # Decoder
    conv_decoder = Sequential([
        get_3d_conv_transpose_layer(64, strides=(1, 4, 4)),
        BatchNormalization(),
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(64),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(32),
        BatchNormalization(),
        get_3d_conv_transpose_layer(3, activation="sigmoid")])

    conv_autoencoder = Sequential([conv_encoder, conv_decoder])

    return conv_autoencoder


def get_2d_convolutional_stack(filters, strides=(2, 2), **kwargs):
    return [get_2d_conv_layer(filters, strides=strides, **kwargs),
            BatchNormalization(),
            get_2d_conv_layer(filters, kernel_size=3, strides=1, activation=None),
            BatchNormalization(),
            Activation("sigmoid")]


def get_2d_deconv_layer(filters, input_tensor, kernel_size=(4, 4), strides=(2, 2), padding="same"):
    return BatchNormalization()(get_2d_conv_transpose_layer(
        filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor))


def get_2d_up_sample_layer(filters, input_tensor, kernel_size=(4, 4), strides=(2, 2)):
    return Activation("sigmoid")(get_2d_conv_transpose_layer(
        filters, kernel_size=kernel_size, strides=strides, activation=None)
                                 (input_tensor))


def dynamics_autoencoder_def_2d_updated(input_shape=(None, 128, 3)):
    """
    definiton for 2d dynamics prediction autoencoder.
    Latent space reduced by a factor of 2 compared to original space.
    Model architecture used by Chen at al. 2022

    Parameters
    ----------
    input_shape : int tuple, optional
        fixes input shape so that summary function of model gives more information.
        The default is (None,None, None, 3).

    Returns
    -------
    conv_autoencoder : tf.keras.Model
        untrained dynamics prediction autoencoder.

    """

    # Encoder
    encoder = Sequential([
        *get_2d_convolutional_stack(32, input_shape=input_shape),
        *get_2d_convolutional_stack(32),
        *get_2d_convolutional_stack(64),
        *get_2d_convolutional_stack(128),
        *get_2d_convolutional_stack(128, strides=(2, 1))])

    input_ = tf.keras.Input(shape=(8, 8, 128))
    output0 = get_2d_deconv_layer(64, input_, strides=(2, 1), kernel_size=(3, 4))
    temp_output = get_2d_conv_transpose_layer(
        3, strides=(1, 1), kernel_size=(3, 3))(input_)
    output1 = get_2d_up_sample_layer(3, temp_output, kernel_size=(3, 4), strides=(2, 1))
    concat = tf.concat([output0, output1], axis=3)

    output0 = get_2d_deconv_layer(64, concat, kernel_size=(4, 4))
    temp_output = get_2d_conv_transpose_layer(
        3, strides=(1, 1), kernel_size=(3, 3))(concat)
    output1 = get_2d_up_sample_layer(3, temp_output)
    concat = tf.concat([output0, output1], axis=3)

    output0 = get_2d_deconv_layer(32, concat, kernel_size=(4, 4))
    temp_output = get_2d_conv_transpose_layer(
        3, strides=(1, 1), kernel_size=(3, 3))(concat)
    output1 = get_2d_up_sample_layer(3, temp_output)
    concat = tf.concat([output0, output1], axis=3)

    output0 = get_2d_deconv_layer(16, concat, kernel_size=(4, 4))
    temp_output = get_2d_conv_transpose_layer(
        3, strides=(1, 1), kernel_size=(3, 3))(concat)
    output1 = get_2d_up_sample_layer(3, temp_output)
    concat = tf.concat([output0, output1], axis=3)

    output = get_2d_conv_transpose_layer(
        3, activation="sigmoid")(concat)

    decoder = Model(inputs=[input_], outputs=[output])

    return Sequential([encoder, decoder])


def latent_autoencoder_def(
        neural_state_dim, input_shape=(2, 1, 1, 64)):
    """
    returns an untrained latent reconstruction autoencoder.

    Parameters
    ----------
    neural_state_dim : int
        latent space dimension of autoencoder.
    input_shape : int shape, optional
        shape of latent dimension of corresponding
        dynamics prediction autoencoder.
        The default is (2 ,4 ,4 ,128).

    Returns
    -------
    autoencoder : tf.keras.Model
        untrained dynamics prediction autoencoder.

    """
    encoder = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(neural_state_dim, activation=None)
    ])

    decoder = Sequential([
        Dense(32, activation='relu', input_shape=[neural_state_dim]),
        BatchNormalization(),
        Dropout(0.15),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(128, activation="sigmoid"),
        BatchNormalization(),
        Dropout(0.15),
        Dense(np.prod(input_shape)),
        Reshape(input_shape)
    ])

    autoencoder = Sequential([
        encoder,
        decoder
    ])

    return autoencoder


