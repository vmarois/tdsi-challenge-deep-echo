
# TDSI project - challenge on ultrasound image analysis #

A python project for the analysis of 2D cardiac ultrasound images through the development of deep learning networks. The proposed framework is built on several libraries :
- Numpy
- Pandas
- SciPy
- SimpleITK
& others built-in packages.

The project is developed by :

* vincent.marois@insa-lyon.fr
* camille.louvet@insa-lyon.fr

### How do I get set up? ###

First, clone the repository :

    git clone git@gitlab.in2p3.fr:olivier.bernard/tdsi-challenge-deep-echo.git


Then, install the required python packages:

    sudo cd tdsi-challenge-deep-echo && pip3 install -r requirements.txt


Following, you can run the main `Python` script:

    python3 read-med-files.py


### Who do I talk to? ###

The project administrators are:

* Olivier Bernard <olivier.bernard@creatis.insa-lyon.fr>



### How is structured this project ? ###
We are structuring the project by creating a local `Python` module containing the functions we implemented.
This structure allows :
- For a clearer and shorter main program,
- We are able to gather some functions by their role in the project (data acquisition, preprocessing, visualization etc)
