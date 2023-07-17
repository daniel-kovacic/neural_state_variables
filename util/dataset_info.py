import os
import json
import numpy as np
import random
import math


class DatasetInfo:
    DATA_INFO_DIR_PATH = "../data_info"
    DATA_DIR_PATH = "../data"

    def __init__(self, dataset_name, frames_per_vid, num_of_vids, train_ind=None, val_ind=None, test_ind=None):
        self.dataset_name = dataset_name
        self.frames_per_vid = frames_per_vid
        self.num_of_vids = num_of_vids

        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind

    def get_path(self, hidden=False):
        return os.path.join(DatasetInfo.DATA_DIR_PATH,
                            self.dataset_name if not hidden else self.dataset_name + "_hidden")

    @staticmethod
    def read_from_file(dataset_name):
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


if __name__ == "__main__":
    data_info = DatasetInfo("double_pendulum", 60, 1100)
    data_info.split_dataset()
    data_info.store_info_dict_as_json()
