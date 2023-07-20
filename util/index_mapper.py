import numpy as np
import tensorflow as tf
import os


# extracts, normalises and rescales the images at the given path
@tf.function
def __extract_normalise_image(image_path):
    """
    preprocesses image files

    Parameters
    ----------
    image_path : str
        path to image file.

    Returns
    -------
    image : tf.Tensor
        image as RGB encoded normalised tensor

    """
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def __get_adjacent_frame_paths(indices, number_of_frames, dir_path):
    """
    Parameters
    ----------
    dir_index : tf.Tensor
        dir_index.
    frame_index : tf.Tensor
        index of the first frame.
    dir_path: str
        path to used data directory
    Returns
    -------
    list
        returns a lsit of the path of adjacent image files
        at the given indices.

    """
    dir_index, frame_index = indices.numpy()[0], indices.numpy()[1]
    return tf.constant([f"{bytes.decode(dir_path.numpy())}/{dir_index}/{frame_index + i}.png"
                        for i in range(number_of_frames)], shape=number_of_frames)


@tf.function
def __get_normalized_images(image_paths):
    """
    preprocess a list of frames with given file name

    Parameters
    ----------
    image_paths : str-list
        list of paths to images that should be preprocessed.

    Returns
    -------
    list
        preprocessed frames.

    """
    return tf.map_fn(__extract_normalise_image, image_paths, fn_output_signature=tf.float32)


def get_data_preprocessor(dataset_info, dim=3, num_of_frames=2, hidden_parts=False):
    path = str(dataset_info.get_path())

    if dim != 3:
        def dim_adjust(x):
            return tf.concat(x, 1)  # tf.map_fn(lambda x: tf.squeeze(x, axis=0), x)
    else:
        def dim_adjust(x):
            return tf.concat(x, 0)
    if not hidden_parts:
        def __data_entry_preprocessor(indices):
            """
            loads and preprocesses adjacent frames.
            Appends frames vertically so that 2D-convolution can be used.

            Parameters
            ----------
            dir_index : Tensor
                directory index of the frame files.
            file_index : Tensor
                index of the first frame.

            Returns
            -------
            x0 : tf.Tensor
                input for 2D convolutional dynamics prediction autoencoder.
            x1 : tf.Tensor
                expected output corresponding to input.

            """

            image_paths = tf.py_function(func=__get_adjacent_frame_paths,
                                         inp=[indices, num_of_frames + 1, path],
                                         Tout=tf.string)
            image_paths.set_shape(num_of_frames + 1)
            images = dim_adjust(__get_normalized_images(image_paths))
            if dim != 3:
                images = [images[i] for i in range(num_of_frames + 1)]
            x0 = tf.concat(images[:num_of_frames], 0)
            x1 = tf.concat(images[1:num_of_frames + 1], 0)
            return x0, x1
    else:
        hidden_path = dataset_info.get_path(hidden=True)

        def __data_entry_preprocessor(indices):
            """
            loads and preprocesses adjacent frames.
            Appends frames along a new axis so that 3D-convolution can be used.
            Uses images with hidden parts as input and full images as expected
            output

            Parameters
            ----------
            dir_index : tf.Tensor
                directory index of the frame files.
            file_index : tf.Tensor
                index of the first frame.

            Returns
            -------
            x0 : tf.Tensor
                input for 2D convolutional dynamics prediction autoencoder.
            x1 : tf.Tensor
                expected output corresponding to input.

            """
            tf.py_function(func=__get_adjacent_frame_paths,
                           inp=[indices, num_of_frames + 1, path],
                           Tout=tf.string)
            input_image_paths = tf.py_function(func=__get_adjacent_frame_paths,
                                               inp=[indices, num_of_frames, hidden_path],
                                               Tout=tf.string)
            output_image_paths = tf.py_function(func=__get_adjacent_frame_paths,
                                                inp=[tf.add(indices, tf.constant([0, 1])), num_of_frames, path],
                                                Tout=tf.string)

            input_images = tf.map_fn(
                dim_adjust, __get_normalized_images(input_image_paths), fn_output_signature=tf.float32)
            output_images = tf.map_fn(
                dim_adjust, __get_normalized_images(output_image_paths), fn_output_signature=tf.float32)
            x0 = tf.concat(input_images, 0)
            x1 = tf.concat(output_images, 0)
            return x0, x1
    return __data_entry_preprocessor


class IndexArrayMapper():
    def __init__(self, data, frames_per_vid, number_of_frames=2, hidden_data=None):
        self.data = data
        self.frames_per_vid = frames_per_vid
        self.output_data = hidden_data if hidden_data else self.data
        self.num_of_frames = number_of_frames

    def index_array_mapper_3d(self, indices):
        array_index = self.frames_per_vid * indices[0] + indices[1]
        return (self.data[array_index: array_index + self.num_of_frames],
                self.output_data[array_index + 1: array_index + self.num_of_frames + 1])

    def index_array_mapper_2d(self, indices):
        array_index = self.frames_per_vid * indices[0] + indices[1]
        return (tf.concat([self.data[i] for i in range(array_index, array_index + self.num_of_frames)]),
                tf.concat(
                    [self.output_data[i] for i in range(array_index + 1, array_index + self.num_of_frames + 1)]))

    def latent_array_mapper(self, indices):
        x = self.data[indices[0] * self.frames_per_vid + indices[1]]
        return x, x

    def get_latent_index_array_mapper(self):
        return lambda i: tf.py_function(
            func=self.latent_array_mapper,
            inp=[i],
            Tout=[tf.float32, tf.float32])

    def get_index_array_mapper(self, dim=3):
        if dim == 3:
            return lambda i: tf.py_function(
                func=self.latent_array_mapper,
                inp=[i],
                Tout=[tf.float32, tf.float32])
        else:
            return lambda i: tf.py_function(
                func=self.latent_array_mapper,
                inp=[i],
                Tout=[tf.float32, tf.float32])


def get_array_mapper(data, frames_per_vid, number_of_frames=2, hidden_data=None, dim=3):
    mapper_obj = IndexArrayMapper(data, frames_per_vid, number_of_frames=2, hidden_data=None)
    return mapper_obj.get_index_array_mapper(dim=dim)


def get_latent_array_mapper(data, frames_per_vid):
    mapper_obj = IndexArrayMapper(data, frames_per_vid)
    return mapper_obj.get_latent_index_array_mapper()


def get_latent_memmap(dataset_info, filename, shape=(2, 1, 1, 64)):
    path = os.path.join(dataset_info.get_latent_path(), filename)
    return np.memmap(path, shape=shape, dtype='float32', mode='r')


def get_preprocessed_video_frames(dataset_info):
    image_paths = [os.path.join(dataset_info.get_path(), str(i) + ".png") for i
                   in range(dataset_info.files_per_folder)]
    return __get_normalized_images(image_paths)


def preprocess_specific_images(dataset_info, hidden_parts=False):
    path = str(dataset_info.get_path())

    def __data_entry_preprocessor(indices):
        """
        loads and preprocesses adjacent frames.
        Appends frames vertically so that 2D-convolution can be used.

        Parameters
        ----------
        dir_index : Tensor
            directory index of the frame files.
        file_index : Tensor
            index of the first frame.

        Returns
        -------
        x0 : tf.Tensor
            input for 2D convolutional dynamics prediction autoencoder.
        x1 : tf.Tensor
            expected output corresponding to input.

        """

        image_paths = tf.py_function(func=__get_adjacent_frame_paths,
                                     inp=[indices, 1, path],
                                     Tout=tf.string)
        return __get_normalized_images(image_paths)

    return __data_entry_preprocessor


if __name__ == '__main__':
    from dataset_info import DatasetInfo
    from images_to_arrays_util import images_to_arrays

    dataset_info = DatasetInfo.read_from_file("double_pendulum")

    prep = get_latent_memmap(dataset_info, "points", (dataset_info.get_data_tuples(), 2, 1, 1, 64))
    for i in range(100):
        for j in range(58):
            print(prep((i, j)))
    # print(images_to_arrays(prep, list(range(dataset_info.num_of_vids)[:50]), dataset_info.frames_per_vid)[:5])
