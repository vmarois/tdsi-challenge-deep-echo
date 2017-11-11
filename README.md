
# TDSI project - challenge on ultrasound image analysis #

A python project for the analysis of 2D cardiac ultrasound images through the development of deep learning networks. The proposed framework is built on several libraries :
- Keras
- Scikit-learn & scikit-image
- Numpy & SciPy
- Pandas
- Matplotlib
- SimpleITK
& others built-in packages (full list in requirements.txt).

The project is developed by :

* vincent.marois@insa-lyon.fr
* camille.louvet@insa-lyon.fr

### How do I get set up? ###

First, clone the repository :

git clone git@gitlab.in2p3.fr:olivier.bernard/tdsi-challenge-deep-echo.git


Install the required python packages:

sudo cd tdsi-challenge-deep-echo && pip3 install -r requirements.txt


You are strongly invited to have a look at the  `read-med-files.py` script. This script shows how to use some of the most used methods defined in the local module **deepecho**.  You can execute this command to run this script :

python3 read-med-files.py

The project is  divided in 3 scripts at the moment :
* `data.py` which processes the raw data to produce 2 files, *images.npy* & *targets.npy*. These 2 files contain respectively the input images for the neural network (resized to 96 x 96), and the target features we want to predict, e.g. the center coordinates (x,y) and the main orientation (2 eigenvalues).

* `trainneuralnet.py` defines, creates, compiles & trains a neural network model. After the network has been trained, it is saved to `model.h5`.

* `predict_visualize.py` loads back the model from file, makes a prediction on one given image and plot the loss, the given image & the result.

### Who do I talk to? ###

The project administrators are:

* Olivier Bernard <olivier.bernard@creatis.insa-lyon.fr>

### How is structured this project ? ###

We are structuring the project by creating a local `Python` module  called  **deepecho**, which contains the methods we implemented as we need them. So far, this module is structured in 3  `Python` files, and each file contains some methods relevant with the topic indicated by the  `Python` file name. Hence, in  `preprocessing.py`, you'll find methods to find the center and the main orientation of a region of interest on a given image.

This structure allows :
- For clearer and shorter main scripts,
- We are able to gather some functions by their role in the project (data acquisition, preprocessing, visualization etc)
- Code maintenance and the addition of future features are simplified.
