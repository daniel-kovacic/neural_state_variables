# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:50:37 2023

@author: kovac
"""
import tensorflow as tf
import cv2
import os


def predict_longterm(input_frames, pred_autoencoder, number_of_steps=20, dim=3):
    """
    function used for long term predictions. Repeatedly
    predicts the next timestep and uses it as final frame of input
    for the next predictions

    Parameters
    ----------
    input_frames : tf.Tensor
        inpit frames as needed for the dynamics prediction autoencoder.
    pred_autoencoder : tf.Model
        dynamics prediciton autoencoder used fo long term prediction.
    number_of_steps : int, optional
        number of timesteps. The default is 20.
    dim : int, optional
        defines if 2D or 3D convolution should be used. Only 3D is implemented
        The default is 3.

    Returns
    -------
    result : tf.Tensor list
        list of all predictions.

    """
    result = []

    if dim == 3:
        for _ in range(number_of_steps):
            input_frames = tf.concat(
                [input_frames[1:], pred_autoencoder.predict(tf.expand_dims(input_frames, axis=0))[0, -1:]], 0)
            result = result + [input_frames[-1]]
    else:
        width = 128
        for _ in range(number_of_steps):
            input_frames = tf.concat(
                [input_frames[width:], pred_autoencoder.predict(tf.expand_dims(input_frames, axis=0))[0, -width:]], 0)
            result = result + [input_frames[-width:]]

    return result


def predict_longterm_latent_supported(input_frames, pred_autoencoder, latent_autoencoder, number_of_steps=20, dim=3,
                                      latent_steps=4):
    """
    function used for long term predictions. Repeatedly
    predicts the next timestep and uses it as final frame of input
    for the next predictions. Uses latent reconstruction to project invalid
    input to a low dimensional space and likely transform it back to valid input.
    latent reconstruction is used every latent_steps-time


    Parameters
    ----------
    input_frames : tf.Tensor
        inpit frames as needed for the dynamics prediction autoencoder.
    pred_autoencoder : tf.Model
        dynamics prediciton autoencoder used fo long term prediction.
    latent_autoencoder : tf.Model
        latent reconstruction autoencoder used for stabilization.
    number_of_steps : int, optional
        number of timesteps. The default is 20.
    dim : int, optional
        defines if 2D or 3D convolution should be used. Only 3D is implemented
        The default is 3.

    latent_steps : int, optional
        number of step at which latent reconstruction is used.
        (Every latent_steps steps a the latent reconstruction autoencoder
         is used) The default is 4.

    Returns
    -------
    result : tf.Tensor list
        list of all predictions.

    """

    result = []
    pred_encoder = pred_autoencoder.layers[0]
    pred_decoder = pred_autoencoder.layers[1]
    width = 128

    for i in range(number_of_steps):
        if i % latent_steps == 0:
            latent_pred = pred_encoder.predict(tf.expand_dims(input_frames, axis=0))
            latent_rec = latent_autoencoder.predict(latent_pred)
            pred = pred_decoder.predict(latent_rec)[0, -1 if dim == 3 else -width:]

        else:
            pred = pred_autoencoder.predict(tf.expand_dims(input_frames, axis=0))[0, -1 if dim == 3 else -width:]

        input_frames = tf.concat([input_frames[1:], pred], 0)
        result = result + [input_frames[-1]] if dim==3 else [input_frames[-width:]]
    return result


def get_abs_difference(predictions, expected):
    return [tf.math.abs(pred - exp) for (pred, exp) in zip(predictions, expected)]


def get_frames_from_single_vid(mapper, video_index, num_frames_per_vid):
    [mapper(video_index, j) for j in range(num_frames_per_vid)]


def store_video_frames(video_frames, path):
    for i, frame in enumerate(video_frames):
        tf.keras.preprocessing.array_to_img(frame * 255, scale=False).save(os.path.join(path, f"{i}.png"))


def combine_frames_to_video(dir_path, video_path):
    """
    converts a set of frames stored in png format to one mp4 video.
    frames must contain end with their cordinal number
    in the following way: "0.png, 1.png... n.png"
    Parameters
    ----------
    dir_path : str
        path to the directory where the video frames are stored.
    video_path : str
        path where the resulting video is stored.
        Needs to include filename and needs to end with .mp4.

    Returns
    -------
    None.

    """

    frames = [frame for frame in os.listdir(dir_path)]
    frames.sort(key=lambda f: int(f[: f.rfind(".")]))
    frames = [os.path.join(dir_path, frame) for frame in os.listdir(dir_path)]

    height, width, layers = cv2.imread(frames[0]).shape
    video_creator = cv2.VideoWriter(video_path, 0, 5, (width, height))

    for frame in frames:
        video_creator.write(cv2.imread(frame))

    cv2.destroyAllWindows()
    video_creator.release()
