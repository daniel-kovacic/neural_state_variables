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
    indices : tf.Tensor
        vid- and frame- indices.
    number_of_frames: int
        number adjacent frames which paths should be returned
    dir_path: str
        path to used data directory
    Returns
    -------
    list
        returns a list of the paths of adjacent image files
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


def get_data_preprocessor(dataset_info, hidden_parts=False):
    """
    Returns data preprocessor corresponding to dataset info object.
    Parameters
    ----------
    dataset_info: DatasetInfo
        dataset info object including DatasetSpecificInfo
    hidden_parts: boolean
        determines if data is split in hidden and non-hidden parts

    Returns
    -------

    """
    path = str(dataset_info.get_path())
    num_of_frames = dataset_info.get_num_of_frames()
    dim = dataset_info.get_dim
    if dim != 3:
        def dim_adjust(x):
            return tf.concat(x, 1)  # tf.map_fn(lambda x: tf.squeeze(x, axis=0), x)
    else:
        def dim_adjust(x):
            return tf.concat(x, 0)
    if not hidden_parts:
        def __data_entry_preprocessor(indices):
            """
            loads and preprocesses adjacent frames and appends them appropriately depending on dim of the dataset.

            Parameters
            ----------
            indices : Tensor
                video and frame index of data that should be preprocessed.

            Returns
            -------
            x0 : tf.Tensor
                input for dynamics prediction autoencoder.
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
            loads and preprocesses adjacent frames including of a dataset, including a hidden part,
             and appends them appropriately depending on dim of the dataset.

            Parameters
            ----------
            indices : Tensor
                video and frame index of data that should be preprocessed.
            Returns
            -------
            x0 : tf.Tensor
                input for dynamics prediction autoencoder.
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


class IndexArrayMapper:
    def __init__(self, data, dataset_info, hidden_data=None):
        """

        Parameters
        ----------
        data: np.array
            data array
        dataset_info: DatasetInfo
            DatasetInfo object containing information about the dataset
        hidden_data: np.array, optional
            optional data array containing data with parts hidden which should be used as input for dynamics prediction
            autoencoder.
        """
        self.data = data
        self.frames_per_vid = dataset_info.frames_per_vid
        self.output_data = hidden_data if hidden_data is not None else self.data
        self.num_of_frames = dataset_info.get_num_of_frames()

    def index_array_mapper_3d(self, indices):
        """

        Parameters
        ----------
        indices: tuple-int
            video index and frame index

        Returns
        -------
            Frames appended on the time axis
        """
        return (self.data[indices[0], indices[1]:indices[1] + self.num_of_frames],
                self.output_data[indices[0], indices[1] + 1: indices[1] + self.num_of_frames + 1])

    def index_array_mapper_2d(self, indices):
        """
        Parameters
        ----------
        indices: tuple-int
            video index and frame index

        Returns
        -------
            input and expected output frames appended on the x-axis
        """
        return (tf.concat([frame for frame in self.data[indices[0], indices[1]:indices[1] + self.num_of_frames]], 0),
                tf.concat([frame for frame in self.output_data[indices[0], indices[1] + 1: indices[1] + self.num_of_frames + 1]], 0))

    def latent_array_mapper(self, indices):
        """
        Parameters
        ----------
        indices: tuple-int
            video index and frame index

        Returns
        -------
            input and expected output frames appended on the x-axis
        """
        x = self.data[indices[0], indices[1]]
        return x, x

    def get_latent_index_array_mapper(self):
        """

        Returns
        -------
        function:
            latent array mapper
        """
        return lambda i: tf.py_function(
            func=self.latent_array_mapper,
            inp=[i],
            Tout=[tf.float32, tf.float32])

    def get_index_array_mapper(self, dim=3):
        """

        Parameters
        ----------
        dim: int, optional
            defines in which way the frames are appended 2 (x-axis) or 3(time-axis)
        Returns
        -------

        """
        if dim == 3:
            return lambda i: tf.py_function(
                func=self.index_array_mapper_3d,
                inp=[i],
                Tout=[tf.float32, tf.float32])
        else:
            return lambda i: tf.py_function(
                func=self.index_array_mapper_2d,
                inp=[i],
                Tout=[tf.float32, tf.float32])


def get_array_mapper(data, data_info, hidden_data=None, latent=False):
    """
    Get mapping from indices to data in np.array
    Parameters
    ----------
    data: np.array
        data to which should be mapped
    data_info: DataInfo
        information about the dataset
    hidden_data: np.array/None, optional
        optonal data with parts hidden, used as input
    latent: boolean, optional
        defines if mapping is to latent encoded data

    Returns
    -------
        mapper from indices to data
    """
    mapper_obj = IndexArrayMapper(data, data_info, hidden_data=hidden_data)
    return mapper_obj.get_latent_index_array_mapper() if latent else mapper_obj.get_index_array_mapper(
        dim=data_info.get_dim())


def get_latent_memmap(dataset_info, filename):
    """
    Get mapping from indices to data in np.array
    Parameters
    ----------
    dataset_info: DataInfo
        information about the dataset
    filename: boolean, str
        filename to memmap file

    Returns
    -------
        mapper from indices to memmap-data
    """
    path = os.path.join(dataset_info.get_latent_path(), filename)
    shape = (dataset_info.number_of_vids, dataset_info.get_data_tuples_per_vid(),
             dataset_info.get_num_of_frames(), dataset_info.get_latent_enc_shape())
    return np.memmap(path, shape=shape, dtype='float32', mode='r')


def get_specific_image_preprocessor(dataset_info, hidden=False):
    """

    Parameters
    ----------
    dataset_info: DatasetInfo
        Object containing information about  the dataset
    hidden: boolean
        determines if a dataset with parts hidden should be used

    Returns
    -------

    """
    path = str(dataset_info.get_path(hidden=hidden))

    def __data_entry_preprocessor(indices):
        """
        loads and preprocesses adjacent frames including of a dataset, including a hidden part,
         and appends them appropriately depending on dim of the dataset.

        Parameters
        ----------
        indices : Tensor
            video and frame index of data that should be preprocessed.
        Returns
        -------
        x0 : tf.Tensor
            input for dynamics prediction autoencoder.
        x1 : tf.Tensor
            expected output corresponding to input.

        """

        image_paths = tf.py_function(func=__get_adjacent_frame_paths,
                                     inp=[indices, 1, path],
                                     Tout=tf.string)
        return __get_normalized_images(image_paths)

    return __data_entry_preprocessor


if __name__ == '__main__':
    pass
