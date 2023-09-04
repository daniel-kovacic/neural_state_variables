# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# plot an illustration of prediction and error on a random element
# of given 2d-dataset given
class SingleStepVisualizationUtil:
    """
    Class with functions for producing plots for visualization 
    of learning progress

    Methods
    -------

    """


    @staticmethod
    def _delta_t_string_helper(i):
        return r" + {}$\Delta t$".format(i if i != 1 else "") if i != 0 else ""

    @staticmethod
    def _split_2d_tensor(tensor):
        width= 128
        print(tf.concat([tf.expand_dims(tensor[i*width:(i* + 1)*width], 0) for i in range(tf.shape(tensor)[0] // width)], 0))
        return tf.concat([tf.expand_dims(tensor[i*width:(i + 1)*width], 0) for i in range(tf.shape(tensor)[0] // width)], 0)

    """
    @staticmethod
    def create_image_tensors_2d(input_image_tensor, prediction_image_tensor, expected_output_image_tensor):

        input_tensor_split = SingleStepVisualizationUtil._split_2d_tensor(input_image_tensor)
        expected_output_tensor_split = SingleStepVisualizationUtil._split_2d_tensor(expected_output_image_tensor)
        prediction_tensor_split = SingleStepVisualizationUtil._split_2d_tensor(prediction_image_tensor)
        error_img = SingleStepVisualizationUtil._split_2d_tensor(
            prediction_image_tensor - expected_output_image_tensor)
        difference_input_expected_output = SingleStepVisualizationUtil._split_to_images_2d(
            prediction_image_tensor - input_image_tensor)
        return [input_tensor_split, expected_output_tensor_split,
                expected_output_tensor_split, prediction_tensor_split,
                error_img, difference_input_expected_output]
    """

    @staticmethod
    def show_single_prediction(autoencoder, dataset, images=None, dim=3, title=None, multiple_plots=False,
                               with_titles=True, save_path=None, latent_rec_autoencoder=None):
        """
        visualizes the accuracy of a single prediction uing a dynamics
        prediction autoencoder which uses 3D-convolution.
        If the images argument != None than it is used as input for the model
        otherwise the dataset argument is used to obtain frames.

        Parameters
        ----------
        autoencoder : tf.Model
            autoencoder which is used for the prediction.
        dataset : tf.data.Dataset
            dataset used to obtain frames. Ignored if images !=None
        title : str, optional
            title of the created plot. The default is None.
        images : tf.Tensor, optional
            images of the shape (batch_size, *input_shape).
            If set to None dataset is used to obtain images.
            The default is None.

        Returns
        -------
        None.

        """

        if not images:
            images = next(iter(dataset))
        if latent_rec_autoencoder:
            SingleStepVisualizationUtil.create_tensors_latent_rec(images, autoencoder, latent_rec_autoencoder)
        input_tensor, output_tensor, prediction_tensor = \
            SingleStepVisualizationUtil.create_image_tensors(images, autoencoder)
        if dim != 3:
            input_tensor = SingleStepVisualizationUtil._split_2d_tensor(input_tensor)
            output_tensor = SingleStepVisualizationUtil._split_2d_tensor(output_tensor)
            prediction_tensor = SingleStepVisualizationUtil._split_2d_tensor(prediction_tensor)

        (input_images, expected_output_images, output_images, error_images, difference_input_expected_output) = \
            SingleStepVisualizationUtil.create_single_prediction_images(
                input_tensor, output_tensor, prediction_tensor)

        if multiple_plots:
            SingleStepVisualizationUtil._plot_predictions_on_multiple_plots(
                input_images, expected_output_images, output_images, error_images, difference_input_expected_output,
                with_titles, save_path=save_path)
        else:
            SingleStepVisualizationUtil._plot_predictions_on_single_plot(
                input_images, expected_output_images, output_images, error_images, difference_input_expected_output,
                title)

    @staticmethod
    def create_single_prediction_images(input_image_tensor, output_image_tensor, prediction_tensor):
        input_images = [tf.keras.preprocessing.image.array_to_img(input_image * 255, scale=False)
                        for input_image in input_image_tensor]
        expected_output_images = [
            tf.keras.preprocessing.image.array_to_img(output_image * 255, scale=False)
            for output_image in output_image_tensor]
        output_images = [tf.keras.preprocessing.image.array_to_img(output_image * 255, scale=False) for
                         output_image in prediction_tensor]
        error_images = [
            tf.keras.preprocessing.image.array_to_img(tf.math.abs(output_image) * 255, scale=False)
            for output_image
            in (prediction_tensor - output_image_tensor)]
        difference_input_expected_output = [
            tf.keras.preprocessing.image.array_to_img(tf.math.abs(output_image) * 255, scale=False)
            for output_image
            in (tf.convert_to_tensor(input_image_tensor) - output_image_tensor)]
        return input_images, expected_output_images, output_images, error_images, difference_input_expected_output

    @staticmethod
    def create_image_tensors(images, autoencoder):
        input_tensor, output_tensor = images[0][0], images[1][0]
        input_batch = tf.expand_dims(input_tensor, axis=0)
        prediction_tensor = autoencoder.predict(input_batch)[0]
        return input_tensor, output_tensor, prediction_tensor

    @staticmethod
    def _plot_predictions_on_single_plot(input_images, expected_output_images, output_images,
                                         error_images, difference_input_expected_output, title):

        number_of_frames = len(input_images)
        fig = plt.figure(figsize=(4 * number_of_frames, 30))

        gs = fig.add_gridspec(5, number_of_frames)
        ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(number_of_frames)] for i in range(5)])

        for a in ax.flat:
            a.axis("off")

        for i in range(0, number_of_frames):
            ax[0][i].imshow(input_images[i])
            ax[0][i].set_title('Input image time t {}'.format(SingleStepVisualizationUtil._delta_t_string_helper(i)))

        for i in range(0, number_of_frames - 1):
            ax[1][i].imshow(expected_output_images[i])
            ax[1][i].set_title('Expected reconstructed image')
        ax[1][number_of_frames - 1].imshow(expected_output_images[number_of_frames - 1])
        ax[1][number_of_frames - 1].set_title('Expected next image')

        for i in range(0, number_of_frames - 1):
            ax[2][i].imshow(output_images[i])
            ax[2][i].set_title('Reconstructed image')
        ax[2][number_of_frames - 1].imshow(output_images[number_of_frames - 1])
        ax[2][number_of_frames - 1].set_title('Predicted image')

        for i in range(0, number_of_frames - 1):
            ax[3][i].imshow(error_images[0])
            ax[3][i].set_title('Reconstruction error bitmap')
        ax[3][number_of_frames - 1].imshow(error_images[1])
        ax[3][number_of_frames - 1].set_title('Prediction error bitmap')

        for i in range(0, number_of_frames):
            ax[4][i].imshow(difference_input_expected_output[i])
            ax[4][i].set_title(
                '\'true\' difference at t{} and t{}'
                .format(SingleStepVisualizationUtil._delta_t_string_helper(i),
                        SingleStepVisualizationUtil._delta_t_string_helper(i + 1)))

        if title:
            fig.suptitle(title)

        plt.show()

    @staticmethod
    def _remove_borders(fig):
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

    @staticmethod
    def _plot_predictions_on_multiple_plots(
            input_images, expected_output_images, output_images, error_images, difference_input_expected_output,
            with_titles=True, save_path=None):

        number_of_frames = len(input_images)
        for i in range(0, number_of_frames):
            fig = plt.figure(frameon=False)
            plt.imshow(input_images[i], aspect="auto")
            SingleStepVisualizationUtil._remove_borders(fig)
            plt.axis("off")
            if with_titles:
                plt.title('Input image time t {}'.format(SingleStepVisualizationUtil._delta_t_string_helper(i)))
            if save_path:
                input_images[i].save("{}_input_{}.png".format(save_path, i), "PNG")
            plt.show()

        for i in range(0, number_of_frames - 1):
            fig = plt.figure(frameon=False)
            plt.imshow(expected_output_images[i], aspect="auto")
            SingleStepVisualizationUtil._remove_borders(fig)
            if with_titles:
                plt.title('Expected reconstructed image')
            plt.axis("off")
            if save_path:
                expected_output_images[i].save(
                    "{}_expected_reconstruction_{}.png".format(save_path, i), "PNG")
            plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(expected_output_images[-1], aspect="auto")
        SingleStepVisualizationUtil._remove_borders(fig)
        plt.axis("off")
        if save_path:
            expected_output_images[-1].save("{}_expected_prediction.png".format(save_path), "PNG")
        if with_titles:
            plt.title('Expected predicted image')
        plt.show()

        for i in range(0, number_of_frames - 1):
            fig = plt.figure(frameon=False)
            plt.imshow(output_images[i], aspect="auto")
            SingleStepVisualizationUtil._remove_borders(fig)
            if with_titles:
                plt.title('Reconstructed image')
            plt.axis("off")
            if save_path:
                output_images[i].save(
                    "{}_reconstruction_{}.png".format(save_path, i), "PNG")
            plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(output_images[number_of_frames - 1], aspect="auto")
        SingleStepVisualizationUtil._remove_borders(fig)
        plt.axis("off")
        if with_titles:
            plt.title('Predicted image')
        if save_path:
            output_images[number_of_frames - 1].save("{}_prediction.png".format(save_path), "PNG")
        plt.show()

        for i in range(0, number_of_frames - 1):
            fig = plt.figure(frameon=False)
            plt.imshow(error_images[i], aspect="auto")
            SingleStepVisualizationUtil._remove_borders(fig)
            if with_titles:
                plt.title('Reconstruction error bitmap')
            plt.axis("off")
            if save_path:
                error_images[i].save("{}_reconstruction_error_{}.png".format(save_path, i), "PNG")
            plt.show()
        fig = plt.figure(frameon=False)
        plt.imshow(error_images[number_of_frames - 1], aspect="auto")
        SingleStepVisualizationUtil._remove_borders(fig)
        plt.axis("off")
        if with_titles:
            plt.title('Prediction error bitmap')
        if save_path:
            error_images[number_of_frames - 1].save("{}_prediction_error.png".format(save_path), "PNG")
        plt.show()

        for i in range(0, number_of_frames):
            fig = plt.figure(frameon=False)
            plt.imshow(difference_input_expected_output[i], aspect="auto")
            SingleStepVisualizationUtil._remove_borders(fig)
            if with_titles:
                plt.title(
                    'Difference\'true\' frames at time t{} and t{}'
                    .format(SingleStepVisualizationUtil._delta_t_string_helper(i),
                            SingleStepVisualizationUtil._delta_t_string_helper(i + 1)))
            plt.axis("off")
            if save_path:
                difference_input_expected_output[i].save("{}_change_between_frames_{}.png".format(save_path, i))
            plt.show()

    @staticmethod
    def create_tensors_latent_rec(
            images, dynamics_pred_autoencoder, latent_rec_autoencoder):
        """
        generates tensors needed for visualization of latent rec.

        Parameters
        ----------
        images : tf.Tensor tuple
           tuple of input and predicted output.
        dynamics_pred_autoencoder : tf.Model
            dynamics predictions autoencoder which the latent reconstruction
            autoencoder is based on.
        latent_rec_autoencoder : tf.Model
            latent reconstruction autoencode which the prediction is based upon.

        Returns
        -------
        input_images : tf.Tensor
            input image tensor.
        expected_output_images : tf.Tensor
            expected output image tensor.
        output_images : tf.Tensor
            actual output image tensor.
        error_images : tf.Tensor
            Difference between input and output
        difference_input_expected_output : tf.Tensor
            difference between consecutive frames.

        """
        dynamics_pred_encoder = dynamics_pred_autoencoder.layers[0]
        dynamics_pred_decoder = dynamics_pred_autoencoder.layers[1]
        input_image_tensor = images[0][0]
        output_image_tensor = images[1][0]
        input_image_batch = tf.expand_dims(input_image_tensor, axis=0)
        latent_representation = dynamics_pred_encoder.predict(input_image_batch)
        latent_reconstruction = latent_rec_autoencoder.predict(latent_representation)
        input_image_prediction = dynamics_pred_decoder.predict(latent_reconstruction)[0]

        input_images = [input_images * 255 for input_images in input_image_tensor]
        expected_output_images = [output_image * 255 for output_image in output_image_tensor]
        output_images = [output_image * 255 for output_image in input_image_prediction]
        error_images = [tf.math.abs(output_image) * 255 for output_image in
                        (input_image_prediction - output_image_tensor)]
        difference_input_expected_output = [tf.math.abs(output_image) * 255 for output_image
                                            in (tf.convert_to_tensor(input_image_tensor) - output_image_tensor)]
        return (input_images, expected_output_images, output_images,
                error_images, difference_input_expected_output)

    @staticmethod
    def show_multiple_predictions(predictions, title=None, as_multiple_plots=False, save_path=None):
        """
        creates plots from images tensors

        Parameters
        ----------
        predictions : tf.Tensor list
            images tensors that should be plotted.
        title : str, optional
            title of the produced plot. The default is None.
        as_multiple_plots : boolean, optional
            defines if all the images should be plotted seperately
            or on a single plot. The default is False.
        Returns
        -------
        None.

        """
        fig = plt.figure(frameon=False)
        number_of_predictions = len(predictions)
        if as_multiple_plots:
            for i, pred in enumerate(predictions):
                plt.axis('off')
                SingleStepVisualizationUtil._remove_borders(fig)
                prediction_img = tf.keras.preprocessing.image.array_to_img(pred * 255, scale=False)
                plt.imshow(prediction_img, aspect='auto')
                if save_path:
                    prediction_img.save("{}/{}.png".format(save_path, i), "PNG")
        else:
            fig, ax = plt.subplots(number_of_predictions, 1, figsize=(100, number_of_predictions))
            for i, pred in enumerate(predictions):
                ax[i].axis('off')
                ax[i].imshow(tf.keras.preprocessing.image.array_to_img(pred * 255, scale=False))
                if title:
                    fig.suptitle(title)

            plt.show()

    def plot_longer_prediction_mse(predictions, expected, title=None):
        """
        plots mse of two listss of image tensors with equal lengths

        Parameters
        ----------
        predictions : tf.Tensor list
        expected : tf.Tensor list
        title : str, optional
            title of the produced plot. The default is None.

        Returns
        -------
        None.

        """
        errors = [tf.reduce_mean(tf.square(pred - exp)).numpy() for (pred, exp) in zip(predictions, expected)]
        plt.plot(range(len(errors)), errors)
        if title:
            plt.title(title)
        plt.show()


if  __name__ == "__main__":
    from dataset_info_util import DatasetInfo
    from dataset_util import DatasetUtil
    from index_mapper import *
    dataset_info = DatasetInfo.read_from_file("double_pendulum")
    dataset_util = DatasetUtil(dataset_info)
    dataset = dataset_util.get_dataset(index_mapper=get_data_preprocessor(dataset_info, dim=3), mode="val",
                                       data_tuples_per_vid=dataset_info.get_data_tuples_per_vid(2))
    autoencoder = tf.keras.models.load_model(
        r"C:\Users\kovac\PycharmProjects\neural_state_variables\models\double_pendulum_dyn_pred_3d_2frames")
    latent_autoencoder = tf.keras.models.load_model(
        r"C:\Users\kovac\PycharmProjects\neural_state_variables\models\double_pendulum_latent_rec_3d_2frames")
    SingleStepVisualizationUtil.show_single_prediction(autoencoder, dataset, dim=3, latent_rec_autoencoder=latent_autoencoder)

