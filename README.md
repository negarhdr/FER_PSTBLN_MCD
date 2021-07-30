# PST-BLN
The implementation of Progressive Spatio-Temporal Bilinear Network (PST-BLN) with Monte Carlo Dropout for Landmark-based Facial Expression Recognition with Uncertainty Estimation, MMSP 2021.
The code is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN).

# Data Preparation

#### Data preparation  
  Download the [AFEW](https://cs.anu.edu.au/few/AFEW.html) [[1]](https://www.computer.org/csdl/magazine/mu/2012/03/mmu2012030034/13rRUxjQyrW), [[2]](https://dl.acm.org/doi/abs/10.1145/2663204.2666275), and [CK+](https://www.pitt.edu/~emotion/ck-spread.htm) [[3]](https://ieeexplore.ieee.org/abstract/document/5543262), [[4]](https://ieeexplore.ieee.org/abstract/document/840611) and [Oulu-CASIA](https://www.oulu.fi/cmvs/node/41316) [[5]](https://www.sciencedirect.com/science/article/pii/S0262885611000515) datasets. 
  In order to extract facial landmarks from the images, you need to download a pretrained landmark extractor model. 
  We used Dlib's landmark extractor which can be downloaded from [here](http://dlib.net/face_landmark_detection.py.html). 
  Please note that these datasets and the landmark extractor model cannot be used for any commercial purposes. 
  
  ##### AFEW data preparation:  
  AFEW dataset consists of a set of video clips collected from movies with actively moving faces in different illumination and environmental conditions. 
  The following preprocessing steps are needed to generate the appropriate landmark data for the method. 
  - Convert the .avi videos to .mp4 format and then extract the video frames using the following function:
  
  ```python
  from FER_PSTBLN_MCD.datasets import frame_extractor
  python frame_extractor.py --video_folder ./data/AFEW_videos/ --frames_folder ./data/AFEW/
  ```
  You need to specify the path of the videos as `--video_folder` and the path of the extracted frames data as `--frames_folder`. 
  
  - Place the downloaded landmark-extractor model in the data directory and extract the facial landmarks from the extracted frames by running the following script:
  
  ```python
  from FER_PSTBLN_MCD.datasets import landmark_extractor
  python landmark_extractor.py --dataset_name AFEW --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/AFEW/ --landmark_folder ./data/AFEW_landmarks/
  ```
  You need to specify the path of the landmark extractor as `--shape_predictor` and the path of the extracted frames and extracted landmarks as `--frames_folder` and `--landmark_folder`. 

 - After extracting the facial landmarks for each category, run the following script for data preprocessing and augmentation: 
  
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import AFEW_data_gen
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import data_augmentation
  python AFEW_data_gen.py --landmark_folder  ./data/AFEW_landmarks/ --data_folder ./data/AFEW_data/ 
  python data_aumentation.py --data_folder ./data/AFEW_data/ --aug_data_folder ./data/AFEW_aug_data/
  ```
  The preprocessed augmented data will be saved in the `--aug_data_folder` path. 
  After generating the preprocessed facial landmark data, generate the facial muscle data as follows:
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import gen_facial_muscles_data
  python gen_facial_muscles_data.py --dataset_name AFEW --landmark_data_folder ./data/AFEW_aug_data/ --muscle_data_folder ./data/muscle_data/
  ```
  
  
##### CK+ data preparation:  
  CK+ dataset consists of a set of image sequences starting from a neutral expression to peak expression and the expressions are performed by different subjects. 
  We select the first frame and the last three frames (including the peak expression) of each sequence for landmark extraction.
  In this dataset, only a subset of image sequences are labeled. 
  The first step in the data preparation is to separate the labeled data for each subject, and place each sample in a folder named by its class label. 
  - Extract the facial landmarks and generate the preprocessed train and test data for 10-fold cross validation using the following script:

  ```python
  from FER_PSTBLN_MCD.datasets import CASIA_CK+_data_gen
  from FER_PSTBLN_MCD.datasets import landmark_extractor
  python landmark_extractor.py --dataset_name CK+ --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/CK+/ --landmark_folder ./data/CK+_landmarks/
  python CASIA_CK+_data_gen.py --dataset_name CK+ --landmark_folder  ./data/CK+_landmarks/ --output_folder ./data/CK+_10fold/
  ```
  - After generating the preprocessed facial landmark data, generate the facial muscle data as follows:
  ```python
  from FER_PSTBLN_MCD.datasets import gen_facial_muscles_data
  python gen_facial_muscles_data.py --dataset_name CK+ --landmark_data_folder ./data/CK+_10fold/ --muscle_data_folder ./data/muscle_data/
  ```

##### Oulu-CASIA data preparation:  
  Oulu-CASIA dataset also consists of a set of image sequences starting from a neutral expression to peak expression and the expressions are performed by different subjects. 
  We used the image sequences captured by the VIS system under NI illumination and we select the first frame and the last three frames (including the peak expression) of each sequence for landmark extraction. 
  - Extract the facial landmarks and generate the preprocessed train and test data for 10-fold cross validation using the following script:
  ```python
  from FER_PSTBLN_MCD.datasets import CASIA_CK+_data_gen
  from FER_PSTBLN_MCD.datasets import landmark_extractor
  python landmark_extractor.py --dataset_name CASIA --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/CASIA/ --landmark_folder ./data/CASIA_landmarks/
  python CASIA_CK+_data_gen.py --dataset_name CASIA --landmark_folder  ./data/CASIA_landmarks/ --output_folder ./data/CASIA_10fold/
  ```
  - After generating the preprocessed facial landmark data, generate the facial muscle data as follows:
  ```python
  from FER_PSTBLN_MCD.datasets import gen_facial_muscles_data
  python gen_facial_muscles_data.py --dataset_name CASIA --landmark_data_folder ./data/CASIA_10fold/ --muscle_data_folder ./data/muscle_data/
  ```
     
# Training & Testing

Change the config file depending on what you want. Two models are developed, PST-BLN and ST-BLN. 
You need to specify the model name in config. 

To find an optimized network topology using the PST-BLN algorithm, run:

    `python main.py --config ./config/AFEW/AFEW_pstbln.yaml`

and to train the ST-BLN model with fixed topology, run: 

    `python main.py --config ./config/AFEW/AFEW_stbln.yaml`

You need to specify the number of inference repeats if you wish to use Monte Carlo Dropout for uncertainty estimation. 

To do inference on a trained model, run: 

`python main.py --config ./config/AFEW/test_stbln.yaml`

# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{heidari2021progressive,  
          title     = {Progressive Spatio-Temporal Bilinear Network with Monte Carlo Dropout for Landmark-based Facial Expression Recognition with Uncertainty Estimation},  
          author    = {Heidari, Negar and Iosifidis, Alexandros},  
          booktitle = {MMSP},  
          year      = {2021},  
    }
# Contact
For any questions, feel free to contact: `negar.heidari@ece.au.dk`