import numpy as np
from util.index_mapper import get_specific_image_preprocessor


def images_to_arrays(image_preprocessor, index_list, frames_per_vid, memmap_path=None):
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
    prep = get_specific_image_preprocessor(dataset_info, hidden=hidden_part)
    return images_to_arrays(prep, list(range(dataset_info.num_of_vids)), dataset_info.frames_per_vid,
                            memmap_path=dataset_info.get_memmap_path(hidden=hidden_part))


if __name__ == "__main__":
    from dataset_info_util import DatasetInfo
    standard_images_to_arrays(dataset_info=DatasetInfo.read_from_file("3body_spring_chaotic_updated"), hidden_part=True)