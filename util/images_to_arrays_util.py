import numpy as np


def images_to_arrays(image_preprocessor, index_list, frames_per_vid):
    array_len = len(index_list) * frames_per_vid
    result = np.array([image_preprocessor((index_list[i // frames_per_vid], i % frames_per_vid))
                       for i in range(array_len)])
    return result
