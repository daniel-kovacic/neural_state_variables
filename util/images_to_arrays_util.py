import numpy as np
from util.index_mapper import get_specific_image_preprocessor


def images_to_arrays(image_preprocessor, index_list, frames_per_vid, memmap_path=None):
    """
    Uses a preprocessor from util.index_mapper to preprocess all frames defined by the input parameters
    and returns them as np.array optionally stores it as memmap file.

    Parameters
    ----------
    image_preprocessor: function
        a mapping between indices frame and normalized np.array representation.
        Can be obtained using the util.index_mapper.get_specific_image_preprocessor method
    index_list: list-int
        list of video indices to be transformed
    frames_per_vid: int
        number of frames per vid
    memmap_path: str, optional
        path to which the arrays should be

    Returns
    -------

    """
    shape = (len(index_list), frames_per_vid)
    result = np.array([[image_preprocessor((index_list[i], j))
                       for j in range(shape[1])]for i in range(shape[0])])
    if memmap_path:
        memmap = np.memmap(
            memmap_path,
            dtype='float32',
            mode='w+',
            shape=result.shape)
        memmap[:] = result[:]
        print(memmap[:10])
        return memmap
    return result


def standard_images_to_arrays(dataset_info, hidden_part=False):
    """
    transforms all images of a dataset to arrays and stores them at a standard path defined by the
    DatasetInfo:get_memmap_path method.
    Parameters
    ----------
    dataset_info: DatasetInfo
        DatasetInfo object corresponding to the dataset, which should be transformed
    hidden_part: boolean
        determines if the dataset is stored at the standard-hidden-dataset-path or standard-dataset-path
    Returns
    -------
    np.array:
        transformed frame data
    """
    prep = get_specific_image_preprocessor(dataset_info, hidden=hidden_part)
    return images_to_arrays(prep, list(range(dataset_info.num_of_vids)), dataset_info.frames_per_vid,
                            memmap_path=dataset_info.get_memmap_path(hidden=hidden_part))


if __name__ == "__main__":
    from dataset_info_util import DatasetInfo
    standard_images_to_arrays(dataset_info=DatasetInfo.read_from_file("3body_spring_chaotic_updated"), hidden_part=True)