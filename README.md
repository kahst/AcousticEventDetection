# Acoustic Event Classification Using Convolutional Neural Networks
By [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Hussein Hussein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Etienne Fabian](https://www.intenta.de/en/home.html), [Jan Schloßhauer](https://www.intenta.de/en/home.html), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en), and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)

## Introduction
This code repo complements our submission to the [INFORMATIK 2017 Workshop WS34](https://informatik2017.de/ws34-dlhd/). This is a refined version of our original code described in the paper. We added comments, removed some of the boilerplate code and added testing functionality. If you have any questions or problems running the scripts, don't hesitate to contact us.

Contact:  [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

Please cite the paper in your publications if it helps your research.

<b>You can download the submission here:</b> [2017_INFORMATIK_AED_CNN.pdf](https://box.tu-chemnitz.de/index.php/s/sfW010bbLEsP4Kw) <i>(Unpublished draft version)</i>

## Installation
This is a Thenao/Lasagne implementation in Python for the classification of acoustic events based on deep features. This code is tested using Ubuntu 14.04 LTS but should work with other distributions as well.

First, you need to install Python 2.7 and the CUDA-Toolkit for GPU acceleration. After that, you can clone the project and run the Python package tool PIP to install most of the relevant dependencies:

```
git clone https://github.com/kahst/AcousticEventDetection.git
cd AcousticEventDetection
sudo pip install –r requirements.txt
```

We use OpenCV for image processing; you can install the cv2 package for Python running this command:

```
sudo apt-get install python-opencv
```

Finally, you need to install Theano and Lasagne:
```
sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

You should follow the Lasagne installation instructions for more details: 
http://lasagne.readthedocs.io/en/latest/user/installation.html

## Training
In order to train a model based on your own dataset or any other publicly available dataset (e.g. [UrbanSound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/index.html)) you need to follow some simple steps: First, you need to organize your dataset with subfolders as class labels. Secondly, you need to extract spectrograms from all audio files using the script <b>AED_spec.py</b>. After that, you are ready to train your model. Finally, you can either evaluate a model using <b>AED_eval.py</b> or make predictions for any sound file using <b>AED_test.py</b>.

### Dataset
The training script uses subfolders as class names and you should provide following directory structure:

```
dataset   
¦
+---event1
¦   ¦   file011.wav
¦   ¦   file012.wav
¦   ¦   ...
¦   
+---event2
¦   ¦   file021.wav
¦   ¦   file022.wav
¦   ¦   ...
¦    
+---...
```
 ### Extracting Spectrograms
We decided to use magnitude spectrograms with a resolution of 512x256 pixels, which represent three-second chunks of audio signal. You can generate spectrograms for your sorted dataset with the script <b>AED_spec.py</b>. You can switch to different settings for the spectrograms by editing the file.

Extracting spectrograms might take a while. Eventually, you should end up with a directory containing subfolders named after acoustic events, which we will use as class names during training. 

### Training a Model
You can train your own model using either publicly available training data or your own sound recordings. All you need are spectrograms of the recordings. Before training, you should review the following settings, which you can find in the <b>AED_train.py</b> file:

- `DATASET_PATH` containing the spectrograms (subfolders as class names).

- `MIN_SAMPLES_PER_CLASS` increasing the number of spectrograms per acoustic event class in order to counter class imbalances (Default = -1).

- `MAX_SAMPLES_PER_CLASS` limiting the number of spectrograms per acoustic event class (Default = None - No limit).

- `VAL_SPLIT` which defines the amount of spectrograms in percent you like to use for monitoring the training process (Default = 0.1).

- `MULTI_LABEL` to switch between softmax outputs (False) or sigmoid outputs (True); Activates batch augmentation (multiple targets per spec).

- `IM_SIZE` defining the size of input images, spectrograms will be scaled accordingly (Default = 512x256 pixels).

- `IM_AUGMENTATION` selecting different techniques for dataset augmentation.

- `BATCH_SIZE` defining the number of images per batch; reduce batch size to fit model in less GPU memory (Default = 32).

- `LEARNING_RATE` for scheduling the learning rate; use `LR_DESCENT = True` for linear interpolation and `LR_DESCENT = False` for steps.

- `PRETRAINED_MODEL` if you want to use a pickle file of a previously trained model; set `LOAD_OUTPUT_LAYER = False` if model output size differs (you can download a pre-trained model [here](https://box.tu-chemnitz.de/index.php/s/8vkQqXbUjVWlt5m)).

- `SNAPSHOT_EPOCHS` in order to continuously save model snapshots; select `[-1]` to save after every epoch; the best model params will be saved automatically after training.

There are a lot more options - most should be self-explanatory. If you have any questions regarding the settings or the training process, feel free to contact us. 

<b>Note:</b> In order to keep results reproducible with fixed random seeds, you need to update your <i>.theanorc</i> file with the following lines:

```
[dnn.conv]
algo_bwd_filer=deterministic
algo_bwd_data=deterministic
```

Depending on your GPU, training might take while...

## Evaluation
After training, you can test models and evaluate them on your local validation split. Therefore, you need to adjust the settings in <b>AED_eval.py</b> to match your task. The most important settings are:

- `TEST_DIR` defining the path to your test data. Again, we use subfolders as class labels (Ground Truth). 

- `TRAINED_MODEL` where you specify the pickle file of your pre-trained model and the corresponding model architecture.

- `SPEC_LENGTH` and `SPEC_OVERLAP` which you should choose according to your training data. Increasing the overlap might reduce the prediction error due to more predictions per file.

- `CONFIDENCE_THRESHOLD` and `MAX_PREDICTIONS` can be used to limit the number of predictions returned.

<b>Note:</b> Test data should be organized as training data, subfolders as class names. Feel free to use different ground truth annotations; all you need to do is edit the script accordingly.

## Testing
If you want to make predictions for a single, unlabeled wav-file, you can call the script <b>AED_test.py</b> via the command shell. We provided some example files in the dataset folder. You can use this script as is, no training required. Simply follow these steps:

<b>1. Download pre-trained model:</b>
```
sh model/fetch_model.sh
```
<b>2. Execute script:</b>
```
python AED_test.py --filenames 'dataset/schreien_scream.wav' --modelname 'AED_Example_Run_model.pkl' --overlap 4 --results 5 --confidence 0.01
```
If everything goes well, you should see an output just like this:

```
HANDLING IMPORTS... DONE!
IMPORTING MODEL... DONE!
COMPILING THEANO TEST FUNCTION... DONE! ( 2 s )
TESTING: dataset/schreien_scream.wav
SAMPLE RATE: 44100 TOP PREDICTION(S):
	schreien 99 %
PREDICTION FOR 4 SPECS TOOK 57 ms ( 14 ms/spec ) 
```
<b>Note:</b> You do not need to specify values for overlap, results and confidence – those are optional. You can define a list of wav-files for prediction. To do so, run the script using `--filenames ['file1.wav', file2.wav', ...]`.

This repo might not suffice for real-world applications, but you should be able to adapt the testing script to your specific needs.

We will keep this repo updated and will provide more testing functionality in the future.
