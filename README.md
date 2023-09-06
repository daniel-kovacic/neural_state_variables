<div align='center'>
    <h1><b>Neural State Variables</b></h1>
    <p>This project aims to provide a set of methods for automatically reducing high dimensional video data
    of physical system to minimal full representations(state variables). The framework utilized to achieve this was
    first introduced in a <a href="https://arxiv.org/pdf/2112.10755.pdf"> preprint paper</a> from 2021.
    The codebase with the authors implementation of this framework is available on
    <a href="https://github.com/BoyuanChen/neural-state-variables"> github </a>. Their ideas were adapted to support more broad use cases
    including hiding elements in the input data and using any number of input frames.
 </p>
</div>

![Python](https://badgen.net/badge/Python/3.9/blue)

</div>
---
## **General Structure**
The framework is implemented in the util module. To make the codebase easier to use, a set of higher level functions
is provided in the facade.py file. An examples of how to use them is provided in the example.py file.
Further documentation for both the util and the facade functionality is provided in the code.


---
## **Data**
Due to their size no datasets are provided in this project. Dataset used in the original paper are available on
<a href="https://github.com/BoyuanChen/neural-state-variables"> github </a>

Datasets can be provided in two ways. 
- As video frames
- As numpy arrays

### video frames as input
The expected file structure of video frame used as input is equivalent to the one used in the original implementation
of the framework. This makes it possible to use their datasets without adjustments.
To illustrate the file structure a (windows) path example relative to the project root is given:
    
    ./data/{dataset_name}/{video_index}/{frame_index}.png

Further, for each dataset a json file is required which stores important information like the total number of videos and the
number of frames per video. To help create this dataset-info file functions from util.dataset_info_util submodule can be used.

### numpy arrays as input
Numpy arrays are expected to be rgb-encoded, preprocessed, normalized and have the following shape:

    (number_of_videos, frames_per_video, image_pixel_len, image_pixel_width, 3)

The number of videos and the frames per video can be chosen freely but the image len and image width are both fixed
to 128 pixels. The util.images_to_arrays submodule can be used create np.arrays from video frames and store them
as np.memmap.

---

## 🗒️ **INSTALLATION**

### local installation:

1. clone the repo

```
git clone https://github.com/daniel-kovacic/neural_state_variables.git
```

2. cd into cloned repo

3. install dependencies

```
pip3 install -r requirements.txt
```

4. run the Example

```
python3 ./neural_state_variable_facade/example.py
```

<br />
---
## 💻 **TECHNOLOGIES**

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

<br />
---