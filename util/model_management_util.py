# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:10:06 2023
"""
import os
import tensorflow as tf


class ModelManagementUtil:
    """
    Class used for loading and saving models to enforce chosen
    naming structure.
    """

    DIR = "../models"

    @staticmethod
    def get_full_model_name(dataset_str, dim=3, frames=2, name=None,
                            is_latent=False):
        """
        returns model name according to naming conventions

        Parameters
        ----------
        dataset_str : str
            dataset name.
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        frames : int, optional
            number of frames the model uses. The default is 2.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False.

        Returns
        -------
        str
            full name according to naming convention.

        """
        model_type = "latent_rec" if is_latent else "dyn_pred"
        return "{}{}_{}_{}d_{}frames".format(
            dataset_str, ("_" + name) if name else "",
            model_type, dim, frames)

    @staticmethod
    def get_model_path(dataset_str, dim=3, frames=2, name=None, is_latent=False):
        """
        returns model path according to naming conventions if model is stored
        in DIR.

        Parameters
        ----------
        dataset_str : str
            dataset name.
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        frames : int, optional
            number of frames the model uses. The default is 2.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False.
        Returns
        -------
        str
            full name according to naming convention.

        """
        return os.path.join(ModelManagementUtil.DIR,
                            ModelManagementUtil.get_full_model_name(
                                dataset_str, dim=dim, frames=frames, name=name, is_latent=is_latent))

    @staticmethod
    def load_model(dataset_str, dim=3, frames=2, name=None, is_latent=False):
        """
        returns model saved model if it was stored
        according to naming conventions

        Parameters
        ----------
        dataset_str : str
            dataset name.
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        frames : int, optional
            number of frames the model uses. The default is 2.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False.

        Returns
        -------
        tf.Model
            model specified by arguments.

        """
        return tf.keras.models.load_model(ModelManagementUtil.get_model_path(dataset_str, dim=dim, frames=frames,
                                                                             name=name, is_latent=is_latent))

    @staticmethod
    def save_model(model, dataset_str, dim=3, frames=2, name=None, is_latent=False):
        """
        saves model according to naming conventions

        Parameters
        ----------
        dataset_str : str
            dataset name.
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        frames : int, optional
            number of frames the model uses. The default is 2.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False.

        Returns
        -------
        tf.Model
            model specified by arguments.

        """
        model.save_model(ModelManagementUtil.get_model_path(
            dataset_str, dim=dim, frames=frames, name=name, is_latent=is_latent))

    @staticmethod
    def get_untrained_model_name(dim=3, name=None, is_latent=False):
        """
        get untrained model name according to convention

        Parameters
        ----------
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False

        Returns
        -------
        str
            name for a specified untrained model.

        """
        model_type = "latent_rec" if is_latent else "dyn_pred"
        return "untrained_{}{}_{}d".format(model_type, ("_" + name) if name else "", dim)

    @staticmethod
    def get_untrained_model_path(dim=3, name=None, is_latent=False):
        """
        get untrained model saved according to convention

        Parameters
        ----------
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False

        Returns
        -------
        str
            name for a specified untrained model.

        """
        return os.path.join(ModelManagementUtil.DIR,
                            ModelManagementUtil.get_untrained_model_name(dim=dim, name=name, is_latent=is_latent))

    @staticmethod
    def save_untrained_model(model, dim=3, name=None, is_latent=False):
        """
        save untrained model according to convention

        Parameters
        ----------
        dim : int, optional
            2/3 depending on if the model uses 2D or 3D convolution.
            The default is 3.
        name : str, optional
            extra name of the model. The default is None.
        is_latent : boolean, optional
            defines if model is latent reconstruction or dynamics
            prediction autoencoder. The default is False

        Returns
        -------
        None

        """
        model.save_model(ModelManagementUtil.get_untrained_model_path(dim=dim, name=name, is_latent=is_latent))
