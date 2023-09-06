import numpy as np

import util
import os

PIXEL_LEN = 128


def get_latent_encoding_shape_from_dataset_info(dataset_info):
    """
        DatasetInfo object is used to determine the shape of the dynamics prediction latent encoding
        Parameters
        ----------
        dataset_info : DatasetInfo
            DatasetInfo object which holds the required information to extract the latent encoding shape
        Returns
        -------
        tuple:
            latent encoding shape
        """
    return get_latent_encoding_shape(dataset_info.get_num_of_frames(), dataset_info.get_dim())


def get_latent_encoding_shape(num_of_frames=2, dim=3):
    """
        The shape of the latent encoding is calculated depending on the number of frames used as input and the
        model architecture used.

        Parameters
        ----------
        num_of_frames : int
            umber of frames used as input for the model
        dim : int
            dim of the convolution used by the model
        Returns
        -------
        tuple:
            latent encoding shape
    """

    return (num_of_frames, 1, 1, 64) if dim == 3 else (4 * num_of_frames, 8, 128)


def load_untrained_dynamics_prediction_autoencoder(dataset_info):
    """
        Initializes the correct untrained dynamics prediction autoencoder.
        Depends on if 2- or 3-dimensional convolution was chosen.

        Parameters
        ----------
        dataset_info : DatasetInfo
            DatasetInfo object which holds the required information to infer input shape and model architecture
        Returns
        -------
        tf.keras.Model:
            latent reconstruction autoencoder
    """
    num_of_frames = dataset_info.get_num_of_frames()
    dim = dataset_info.get_dim()
    if dim == 3:
        shape = (num_of_frames, PIXEL_LEN, PIXEL_LEN, 3)
        autoencoder = util.model_definition_util.dynamics_autoencoder_def_3d_updated2(input_shape=shape)
    else:
        shape = (num_of_frames * PIXEL_LEN, PIXEL_LEN, 3)
        autoencoder = util.model_definition_util.dynamics_autoencoder_def_2d(input_shape=shape)

    dataset_info.add_model_specific_info(num_of_frames=num_of_frames, dim=dim,
                                         latent_enc_shape=get_latent_encoding_shape_from_dataset_info(dataset_info))
    return autoencoder


def load_untrained_latent_reconstruction_autoencoder(dataset_info):
    """
    Initializes the required untrained reconstruction autoencoder
    Parameters
    ----------
    dataset_info: DatasetInfo
        DatesetInfo object which holds the information required to initialize the required untrained reconstruction
        autoencoder.

    Returns
    -------
        tf.keras.Model:
            latent reconstruction autoencoder
    """
    shape = get_latent_encoding_shape_from_dataset_info(dataset_info)
    return util.model_definition_util.latent_autoencoder_def(dataset_info.get_neural_state_dim(), input_shape=shape)


def get_dataset_info_from_file(dataset_name):
    """
    reads dataset information from json files saved to ../data_info/{dataset_name}_info.json.

    Parameters
    ----------
    dataset_name: str
        name of the dataset which info should be loaded

    Returns
    -------
    DatasetInfo
    """
    return util.dataset_info_util.DatasetInfo.read_from_file(dataset_name)


def create_dataset_info(dataset_name, frames_per_video, num_of_videos, train_ind=None,
                        val_ind=None, test_ind=None, save_data_info=False, dim=3, num_of_frames=2):
    """
    creates a dataset_info object and optionally persistently stores it as json to ../data_info/{dataset_name}_info.json

    Parameters
    ----------
    dataset_name: str
        name of the dataset that is in use
    frames_per_video: int
        number of frames used as input for the corresponding model
    num_of_videos: int
        number of videos in the dataset
    train_ind: list int
        list of video indices corresponding to videos which should be used for training.
        If it is not specified the partitioning of data in train/val/test is done randomly in a 0.8/0.1/0.1 split
    val_ind: list int
        list of video indices corresponding to videos which should be used for validation.
    test_ind: list int
        list of video indices corresponding to videos which should be used for testing.
    dim : int
        dim of the convolution used by the model
    num_of_frames : int
        number of frames used as input for the model
    save_data_info: boolean
        defines if the created save-data should be saved to a json file.


    Returns
    -------
    DatasetInfo
    """
    dataset_info = util.dataset_info_util.DatasetInfo(
        dataset_name, frames_per_video, num_of_videos, train_ind=train_ind, val_ind=val_ind, test_ind=test_ind)

    if not train_ind:
        dataset_info.split_dataset()
    if save_data_info:
        dataset_info.store_info_dict_as_json()

    dataset_info.add_model_specific_info(num_of_frames=num_of_frames, dim=dim, latent_enc_shape=
    get_latent_encoding_shape(num_of_frames=num_of_frames, dim=dim))
    return dataset_info


def get_memmap_data(file_path, data_info):
    """
    returns np.memmap frames at the file path using the data_info object to determine the shape.
    Parameters
    ----------
    file_path: str
        path where np.memmap frame data is stored
    data_info: DatasetInfo
        DatasetInfo objects containing the information required to determine the shape of the memory mapped data

    Returns
    -------
    np.memmap:
        memmaped numpy arra containing preprocessed data frames.
    """
    shape = (data_info.num_of_vids, data_info.frames_per_vid, PIXEL_LEN, PIXEL_LEN, 3)
    return np.memmap(file_path, shape=shape, dtype='float32')


def get_dataset_generator(dataset_info, data_array=None, hidden_data_array=None, has_hidden_parts=False, latent=False):
    """
    a function which returns a parametrized version of the get_dataset function which allows to obtain
    different dataset for the same model repeatedly specifying all the model/dataset specific data
    Parameters
    ----------
    dataset_info: DatasetInfo
        DatasetInfo objects which obtains important information about dataset and models
    data_array: np.Array
        numpy array containing the input frames
    hidden_data_array: np.array.
    If this is not specified it is assumed that the data is stored as images in the required format.
        numpy array containing the expected output frames
    has_hidden_parts: boolean
        deteremines if the dataset has hidden parts only relevant if no arrays are provided as input
    latent: boolean
        determines if the returned dataset generator is on latent or original frame-data.

    Returns
    -------
        function:
            function which returns datasets without having to specify all model/dataset specific data.

    """

    def parametrized_get_dataset(mode="train", one_vid_per_batch=False, sequential=False):
        return get_dataset(dataset_info, data_array=data_array, hidden_data_array=hidden_data_array,
                           has_hidden_parts=has_hidden_parts, latent=latent, mode=mode,
                           one_vid_per_batch=one_vid_per_batch, sequential=sequential)

    return parametrized_get_dataset


def get_dataset(dataset_info, data_array=None, hidden_data_array=None, has_hidden_parts=False,
                sequential=False, mode="train", latent=False, one_vid_per_batch=False):
    """

    Parameters
    ----------
    dataset_info: DatasetInfo
        DatasetInfo objects which obtains important information about dataset and models
    data_array: np.Array
        numpy array containing the input frames
    hidden_data_array: np.array.
    If this is not specified it is assumed that the data is stored as images in the required format.
        numpy array containing the expected output frames
    has_hidden_parts: boolean
        deteremines if the dataset has hidden parts only relevant if no arrays are provided as input
    latent: boolean
        determines if the returned dataset generator is on latent or original frame-data.
    sequential: boolean
        determines if the returned dataset generates data sequentially or randomly
    mode: str
        string with value: "train", "val" or "test" determines which set of videos the dataset consists of
    one_vid_per_batch: boolean
        determines if each batch should consist of a single video or the default 32 data tuples

    Returns
    -------

    """
    if one_vid_per_batch:
        batch_size = dataset_info.get_data_tuples_per_vid()
    else:
        batch_size = 32
    if data_array is not None:
        index_mapper = util.index_mapper.get_array_mapper(data_array, dataset_info, hidden_data=hidden_data_array,
                                                          latent=latent)
    else:
        index_mapper = util.index_mapper.get_data_preprocessor(dataset_info, hidden_parts=has_hidden_parts)

    dataset_util = util.dataset_util.DatasetUtil(dataset_info, parts_hidden=has_hidden_parts)
    dataset = dataset_util.get_dataset(index_mapper,
                                       data_tuples_per_vid=dataset_info.get_data_tuples_per_vid(),
                                       sequential=sequential,
                                       mode=mode,
                                       batch_size=batch_size)
    return dataset


def train_dynamics_prediction_autoencoder(train_dataset, val_dataset, dyn_pred_autoencoder,
                                          steps_per_epoch=300, validation_steps=30, epochs=100, save_path=None):
    """
    Trains the given dynamics prediction autoencoder. Automatically terminates if the validation error does not improve.
    Optionally saves new best model weights to the specified path after each epoch

    Parameters
    ----------
    train_dataset: tf.data.Dataset
        Dataset consisting of the videos labeld as training data
    val_dataset:
        Dataset consisting of the videos labeld as validation data
    dyn_pred_autoencoder: tf.keras.Model
        dynamics prediction autoencoder which should be trained
    steps_per_epoch: int
        steps per epoch since the dataset used are mostly very big it often time makes sense to not define epochs
        in the standard way
    validation_steps: int
        steps used to calculate the validation error
    epochs: int
        maximum number of epochs before training terminates
    save_path: None/str
        path to which the model weights should be saved after each epoch.

    Returns
    -------

    """
    util.training_util.train_autoencoder(dyn_pred_autoencoder, train_dataset, val_dataset, steps_per_epoch,
                                         validation_steps=validation_steps, epochs=epochs, save_path=save_path)


def train_latent_rec_autoencoder(train_dataset, val_dataset, latent_rec_autoencoder,
                                 steps_per_epoch=1700, validation_steps=100, epochs=500, save_path=None):

    """
    Trains the given dynamics prediction autoencoder. Automatically terminates if the validation error does not improve.
    Optionally saves new best model weights to the specified path after each epoch

    Parameters
    ----------
    train_dataset: tf.data.Dataset
        Dataset consisting of the videos labeld as training data
    val_dataset:
        Dataset consisting of the videos labeld as validation data
    latent_rec_autoencoder: tf.keras.Model
        Latent reconstruction autoencoder which should be trained
    steps_per_epoch: int
        steps per epoch since the dataset used are mostly very big it often time makes sense to not define epochs
        in the standard way
    validation_steps: int
        steps used to calculate the validation error
    epochs: int
        maximum number of epochs before training terminates
    save_path: None/str
        path to which the model weights should be saved after each epoch.

    Returns
    -------

    """
    util.training_util.train_autoencoder(latent_rec_autoencoder, train_dataset, val_dataset, steps_per_epoch,
                                         validation_steps=validation_steps, epochs=epochs, save_path=save_path)


def create_latent_space_predictions(dataset_info, dataset, dynamics_pred_autoencoder, memmap=False, filename="points"):
    """
    Creates the latent space prediction of a dynamics prediction autoencoder for the specified dataset and optionally
    stores them to a np.memmap file at the specified path.

    Parameters
    ----------
    dataset_info: DatasetInfo
        DatasetInfo object with the required infomration about the dataset
    dataset: tf.data.Dataset
        Dataset object holding the data that should be encoded. This Dataset should return all the data from one video
         per batch and should return the data sequentially.
    dynamics_pred_autoencoder: tf.keras.Model
        autoencoder used for encoding the data
    memmap: boolean
        determines if the result should be stored to a file in the as memmap.
    filename:
        the name of the memmap file. full path is ./latent_data/{filename} relative to the project root

    Returns
    -------
    np.array:
        full latent encoding of the shape (num_of_vids, number_of_encodings, *encoding_shape)
        where number_of_encodings = frames_per_vid - number_of_frames + 1
    """
    return util.create_latent_space_predictions.create_encodings(
        dataset_info, dataset, dynamics_pred_autoencoder, memmap=memmap, filename=filename)


def find_intrinsic_dimension(dataset_info, latent_encoding):
    """
    Finds the intrinsic dimension of a latent encoding of the dataset using the Levina Bickel's algorithm.

    Parameters
    ----------
    dataset_info: DatasetInfo
        DatasetInfo object used to determine the shape of the encoding.
    latent_encoding: np.array
        the encodings on which the Levina Bickel's algorithm is used

    Returns
    -------
    int:
        The rounded intrinsic dimension found by the Levina-Bickel's algorithm

    """

    # flattens encoding to the correct shape as the used nearest neighbour implementation of Scikit-learn only
    # accepts standard Matrices as input
    flattend_encoding = util.levina_bickels_algorithm.reshape_array_for_levina_bickels_alg(latent_encoding,
                                                                                           dataset_info)
    intrinsic_dim = util.levina_bickels_algorithm.levina_bickels_alg(flattend_encoding)[0]
    dataset_info.set_neural_state_dim(round(intrinsic_dim))
    return intrinsic_dim


def visualize_single_prediction(autoencoder, dataset, dataset_info, latent_rec_autoencoder=None):
    """
    outputs a visualization of a single prediction and its errors, optionally a dynamics prediction autoencoder and
    combined with its corresponding latent recreation autoencoder can be visualized together

    Parameters
    ----------
    autoencoder: tf.keras.Model
        autoencoder for which a single prediction should be visualized. Either a latent reconstruction autoencoder
        or a dynamics prediction autoencoder.
    dataset: tf.data.Datasets
        dataset from which one prediction will be visualized
    dataset_info: DatasetInfo
        DatasetInfo with the required information
    latent_rec_autoencoder: tf.keras.Model/None
        optional latent_rec_autoencoder which will be visualized in combination with the first autoencoder
        which in this case has to be the corresponding dynamics prediction autoencoder

    Returns
    -------

    """
    title = f"Single Step {'Latent reconstruction ' if latent_rec_autoencoder else ''}prediction"
    util.single_step_visualization_util.SingleStepVisualizationUtil.show_single_prediction(
        autoencoder, dataset, dim=dataset_info.get_dim(), latent_rec_autoencoder=latent_rec_autoencoder, title=title)


def make_longterm_prediction(dataset_info, dynamics_pred_autoencoder, video_index=0, path="./",
                             latent_rec_autoencoder=None):
    """

    Parameters
    ----------
    dataset_info: DatasetInfo
    dynamics_pred_autoencoder: tf.keras.Model
        dynamics prediction autoencoder which should be visualied
    video_index: int
        index of the video which should be visualized
    path: str
        path where the longterm prediction videos should be stored
    latent_rec_autoencoder: tf.keras.Model/None
        optional latent_rec_autoencoder to show latent supported longterm predicitons
    Returns
    -------

    """
    num_of_frames = dataset_info.get_num_of_frames()
    index_mapper = util.index_mapper.get_specific_image_preprocessor(dataset_info)
    single_video_frames = util.longterm_prediction_util.get_frames_from_single_vid(
        index_mapper, video_index, dataset_info.frames_per_vid)
    input_frames = [single_video_frames[i] for i in range(dataset_info.get_num_of_frames())]
    longterm_pred = util.longterm_prediction_util.predict_longterm(
        input_frames, dynamics_pred_autoencoder,
        number_of_steps=dataset_info.frames_per_vid - num_of_frames,
        dim=dataset_info.get_dim())

    if latent_rec_autoencoder:
        latent_supported_longterm_pred = util.longterm_prediction_util.predict_longterm_latent_supported(
            input_frames,
            dynamics_pred_autoencoder,
            latent_rec_autoencoder,
            number_of_steps=dataset_info.frames_per_vid - num_of_frames,
            dim=dataset_info.get_dim())
    names = ["true_frames", "predictions", "latent_supported_predictions"]
    if latent_rec_autoencoder:
        vids_frames = [[frames[0] for frames in input_frames[num_of_frames:]], longterm_pred,
                       latent_supported_longterm_pred]
    else:
        vids_frames = [single_video_frames[num_of_frames:], longterm_pred]
    paths = [os.path.join(path, name) for name in names]
    vid_path = os.path.join(path, "videos")
    os.makedirs(vid_path, exist_ok=True)

    for path, vid_frames, name in zip(paths, vids_frames, names):
        os.makedirs(name, exist_ok=True)
        util.longterm_prediction_util.store_video_frames(vid_frames, path)
        util.longterm_prediction_util.combine_frames_to_video(path, os.path.join(vid_path, name))
