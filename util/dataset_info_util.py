import os
import json
import numpy as np
import random
import math
from util.model_specific_info import ModelSpecificInfo


class DatasetInfo:
    DATA_INFO_DIR_PATH = "../data_info"
    DATA_DIR_PATH = "../data"
    LATENT_DATA_DIR_PATH = "../latent_data"
    NEURAL_DATA_DIR_PATH = "../neural_data"
    MEMMAP_DATA_DIR_PATH = "../memmap_data"

    def __init__(self, dataset_name, frames_per_vid, num_of_vids, train_ind=None, val_ind=None, test_ind=None,
                 model_specific_info=None):
        """

        Parameters
        ----------
        dataset_name:str
        frames_per_vid
        num_of_vids
        train_ind
        val_ind
        test_ind
        model_specific_info
        """

        self.dataset_name = dataset_name
        self.frames_per_vid = frames_per_vid
        self.num_of_vids = num_of_vids
        self.model_specific_info = model_specific_info

        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind

    def get_path(self, hidden=False, general_dir_path=False):
        """
        returns the path to specific dataset corresponding to the object or the path to the general dataset dir

        Parameters
        ----------
        hidden: boolean
            defines if the path is to a dataset where a part is hidden
        general_dir_path: str/None
            decides if the dir path should be returned in which all the datasets are stored


        Returns
        -------
            str:
                dir path to specific dataset/ path to general dataset dir
        """
        if general_dir_path:
            return DatasetInfo.DATA_DIR_PATH
        return os.path.join(DatasetInfo.DATA_DIR_PATH,
                            self.dataset_name if not hidden else self.dataset_name + "_hidden")

    def get_latent_path(self, general_dir_path=False):
        """
        returns the path to specific latent dataset corresponding to the object or the path to the general
        latent dataset dir

        Parameters
        ----------
        general_dir_path: str/None
            decides if the dir path should be returned in which all latent datasets are stored


        Returns
        -------
            str:
                dir path to specific latent dataset/ path to general latent dataset dir
        """
        if general_dir_path:
            return DatasetInfo.LATENT_DATA_DIR_PATH
        return os.path.join(DatasetInfo.LATENT_DATA_DIR_PATH, self.dataset_name)

    def get_neural_path(self, general_dir_path=False):
        """
        returns the path to specific neural dataset corresponding to the object or the path to the general
        neural dataset dir

        Parameters
        ----------
        general_dir_path: str/None
            decides if the dir path should be returned in which all neural datasets are stored

        Returns
        -------
            str:
                dir path to specific neural dataset/ path to general neural dataset dir
        """
        if general_dir_path:
            return DatasetInfo.NEURAL_DATA_DIR_PATH
        return os.path.join(DatasetInfo.NEURAL_DATA_DIR_PATH, self.dataset_name)

    def get_memmap_path(self, general_dir_path=False, hidden=False):
        """
        returns the path to the specific memmap corresponding to the object or the path to the general memmap
        dataset dir

        Parameters
        ----------
        hidden: boolean
            defines if the path is to a dataset where a part is hidden
        general_dir_path: str/None
            decides if the dir path should be returned in which all the datasets are stored

        Returns
        -------
            str:
                dir path to specific dataset/ path to general dataset dir

        """
        if general_dir_path:
            return DatasetInfo.MEMMAP_DATA_DIR_PATH
        return os.path.join(DatasetInfo.MEMMAP_DATA_DIR_PATH, self.dataset_name + ("_hidden" if hidden else ""))

    @staticmethod
    def read_from_file(dataset_name):
        """
        reads dataset info from json file in data_info folder and creates DatasetInfo object.

        Parameters
        ----------
        dataset_name: str
            name of the dataset which's info should be read from json file

        Returns
        -------
            str:
                DatasetInfo created from the json file
        """
        info_path = os.path.join(DatasetInfo.DATA_INFO_DIR_PATH, dataset_name + "_info.json")

        # most relevant information is stored inside the json_file
        with open(info_path, "r") as json_file:
            data_info = json.load(json_file)

        frames_per_vid = data_info["frames_per_video"]
        number_of_vids = data_info["number_of_videos"]

        train_ind = np.array(data_info["training_indices"])
        val_ind = np.array(data_info["validation_indices"])
        test_ind = np.array(data_info["test_indices"])

        return DatasetInfo(dataset_name, frames_per_vid, number_of_vids, train_ind, val_ind, test_ind)

    def store_info_dict_as_json(self):
        """
        stores relevant information about dataset stored in dataset object to json file.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        data_info = dict()
        data_info["frames_per_video"] = self.frames_per_vid
        data_info["number_of_videos"] = self.num_of_vids

        data_info["training_indices"] = self.train_ind
        data_info["validation_indices"] = self.val_ind
        data_info["test_indices"] = self.test_ind

        path = os.path.join(DatasetInfo.DATA_INFO_DIR_PATH, f"{self.dataset_name}_info.json")
        with open(path, "w") as file:
            json.dump(data_info, file, ensure_ascii=False, indent=4)

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """
        randomly splits the datasets videos in the specified ratio

        Parameters
        ----------
        train_ratio : float, optional
            share of the data used for training. The default is 0.8.
        val_ratio : float, optional
            share of the data used for validation. The default is 0.1.

        Returns
        -------
        training_dataset_indices : int list
            indices of the folders used for training.
        validation_dataset_indices : int list
            indices of the folders used for validation.
        test_dataset_indices : int list
            indices of the folders used for testing.

        """
        self.train_ind, self.val_ind, self.test_ind = DatasetInfo.__split_indices(
            self.num_of_vids, train_ratio, val_ratio)

    @staticmethod
    def __split_indices(num_of_vids, train_ratio, val_ratio):
        dataset_indices = list(range(0, num_of_vids))
        random.Random(1).shuffle(dataset_indices)

        val_index = math.floor(num_of_vids * train_ratio)
        test_index = math.floor(num_of_vids * (train_ratio + val_ratio))

        train_ind = dataset_indices[:val_index]
        val_ind = dataset_indices[val_index:test_index]
        test_ind = dataset_indices[test_index:]

        return train_ind, val_ind, test_ind

    def reduce_training_data(self, number_of_points, number_of_frames=2):
        """
        reduces the number of training points to the specified value by setting the number of training folders
        and the number of frames per video to the correct values. To improve the diversity of the
        resulting training set, as many videos as possible are used as input.

        Parameters
        ----------
        number_of_points: int
            number of points in the updated dataset
        number_of_frames: int
            number of input frames, necessary to calculate the number of data points per video

        Returns
        -------

        """
        self.frames_per_vid = (number_of_points // len(self.train_ind) + number_of_frames + 1)
        number_of_training_folders = (number_of_points // (self.frames_per_vid - number_of_frames + 1))
        self.train_ind = self.train_ind[:number_of_training_folders]

    def get_data_tuples_per_vid(self):
        """
        Returns the number of data points per video defined by the number of frames per video and the number
        of frames used as input.
        Returns
        -------
        int:
            number of data points per video
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        return self.frames_per_vid - self.model_specific_info.num_of_frames

    def get_data_tuples(self):
        """
        Returns the total number of data points in the dataset
        Returns
        -------
        int:
            total number of data points in the dataset
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        return self.num_of_vids * self.get_data_tuples_per_vid()

    def add_model_specific_info(self, num_of_frames=2, latent_enc_shape=(2, 1, 1, 64), dim=3):
        """
        Adds a ModelSpecificInfo object to the DatasetInfo object which holds the information which depends
        on the used models.
        Parameters
        ----------
        num_of_frames: int
            number of frames used as input
        latent_enc_shape: tuple
            shape of the latent encoding
        dim: int
            input data shape. Either 3(frames appended on time axis) or 2(frames appended on x-axis)


        Returns
        -------

        """
        self.model_specific_info = ModelSpecificInfo(num_of_frames, latent_enc_shape, dim)

    def get_num_of_frames(self):
        """
        Returns
        -------
        int: number of frames used as input for dynamics prediction autoencoder
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        return self.model_specific_info.num_of_frames

    def get_latent_enc_shape(self):
        """

        Returns
        -------
        tuple: encoding shape of the dynamics prediction autoencoder
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        return self.model_specific_info.latent_enc_shape

    def get_dim(self):
        """

        Returns
        -------
        int: Returns dimension of input frames Either 3(frames appended on time axis) or 2(frames appended on x-axis)
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        return self.model_specific_info.dim

    def get_neural_state_dim(self):
        """

        Returns
        -------
        int: Returns neural state dimension calculated by the Levina Bickel's algorithm
        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        if self.model_specific_info.neural_state_dim is None:
            Exception("No neural_state_dimension set")
        return self.model_specific_info.neural_state_dim

    def set_neural_state_dim(self, neural_state_dim):
        """
        int: Sets neural state dimension calculated by the Levina Bickel's algorithm
        Parameters
        ----------
        neural_state_dim: int
            int: Neural state dimension calculated by the Levina Bickel's algorithm
        Returns
        -------

        """
        if self.model_specific_info is None:
            Exception("No model specific info added so far")
        self.model_specific_info.neural_state_dim = neural_state_dim


if __name__ == "__main__":
    data_info = DatasetInfo("3body_spring_chaotic_updated", 6, 5999)
    data_info.split_dataset()
    data_info.store_info_dict_as_json()
