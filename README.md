# Machine learning function wrapper

## Description
The purpose of this project has been for me to practice Python and Machine Learning
<br />It is a wrapper for trying out Machine Learning functions. 
The project came to be as I tried to learn how to use several Machine Learning models and was tired to write a new script for every single one of them.<br />
Scripts lying allover my computer I decided to create a single repo for them<br />
If you want to test out the project it will require from you some basic knowledge of using the commandline but the instructions are hopefully clear enough for a quick try.<br />
<br />This is mainly just a place for me to store my code, so it is not stable, but everything that is in the guide is tested on Windows. 

## Dependencies
Project has been tested with these dependency versions
 * Python **3.7.x**
<br />You can check if the correct version of the Python is installed by opening commandline (Mac & Linux: Terminal, Windows: Command Prompt or Powershell) and by typing:<br /> `python --version`
**If you're using Mac the python will most likely use Python version 2.7. To check for Python 3 version use `python3 --version`**
If the first numbers in the version are 3.7 it should work. If the first numbers are something else, you can try, **but if your Python is version 2.x.x (2.7.0) it most certainly will not work.**<br />
Official Python downloads page:
<br />https://www.python.org/downloads/ 

 * Pip
<br />Python package installer, which is used to install packages. <br />Official installation guide is quite strait forward:<br />https://pip.pypa.io/en/stable/installing/ but the required steps are also here:<br />
Download via commandline get-pip.py file, which is used to install pip:<br />`$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`<br />After file is installed run:<br />`python get-pip.py` **Or in Mac if python3 returned Python version 3 use** `python3 get-pip.py`

<br />If you want to use data stored in MySQL database.<br /> **If you don't care about MySQL implementation you can skip this part.**
 * Mysql **8.0.20**
 <br />Link to Official MySQL documentation:
 <br />https://dev.mysql.com/doc/refman/8.0/en/installing.html 

# Setup
**After installing Python**

In commandline:

1. Open commanline.<br />If you are using Mac or Linux it is the application called Terminal. If Windows Command Prompt or PowerShell.
2. Move to your working directory. This is the directory where this project is stored.<br />`cd <working directory>`
3. Clone this project in to it. <br />`git clone git@github.com:doslindos/ml_stuff.git`
4. Move to the created project<br />`cd ml_stuff`
5. Create virtualenvironment<br />`virtualenv env` <br />
6. Activate the created virtualenv<br />in **Mac and Linux**<br />`source env/bin/activate`<br />in **Windows**<br />`env\Scripts\activate`
<br />If (env) appeared in the most left corner of the commandline it means that the virtualenv is activated.
7. You can check that your virtualenv python version is the one you want<br />`python --version`
8. **Before installing packages with pip, go to requirements.txt file and uncomment the version of tensorflow you want to use!**<br />If you do not have GPU support installed, use "tensorflow". 
9. If everything is in order, install packages with pip<br />`pip install -r requirements.txt`
<br />If packages were installed without errors installation is done.<br />Now you can tap yourself on the shoulder and jump to the **Quick test guide**!

# Quick test guide

This guide is for quickly trying out some models!
## MNIST
First go to the project directory in commandline (directory where you downloaded the project in the setup (name is ml_stuff))<br />`cd <project directory>`
<br /><br />Next activate the environment (just like in Setup #6)

### Classification
<br /><br />This command is going to download the MNIST dataset in the project folders (stored in data/handlers/mnist/datasets/mnist/), preprocess the images lastly it setups the model and starts to train it.
<br />When the training starts the progress is printed on the terminal.<br />First is **the batch number** <br />with the mnist_basic configurations 50 batches equals the whole training set <br />**validation loss** validation loss tells you the models performance for a set of data (validation set) which it is not seen before. The loss is basically the difference between true labels and predictions so if it goes down, the models predictions are closer to the true lables.<br />**accuracy** (not implemented yet! for now it is empty)
<br /><br />The training command:
<br />`python create.py train -dh mnist -c mnist_basic`
<br /><br />After the training is done you can test how the model performs on a test dataset (data which the model has not seen before).
<br />The testing command:
<br />`python model_tests.py test_model -test classification_test`
<br /><br />First this command will open up a window to select the model to be used.
<br />It will look like this:
<br />![alt text](https://github.com/doslindos/ml_stuff/blob/master/sources/example_images/model_selection.png?raw=true)

<br /><br />After you choose a model (a folder which does have a date as a name) the test dataset inputs are run through the model and the inputs with the models outputs are stored in a file (inside the model folder).
<br />This step is done so that you do not have to run the model everytime you run tests on the same dataset.
<br />After this the confusion matrix is plotted and it looks like this:
<br />![alt text](https://github.com/doslindos/ml_stuff/blob/master/sources/example_images/confusion_matrix.png?raw=true)
<br />In the picture x-axis represents prediction and y-axis the actual label of instances.
<br />For example in the left uppermost corner block is the number of instances the model predicted as zeros which in fact are zeros.
<br />The second block from it to the right is the number of instances the model predicted as ones which were actually zeros.
<br />Red and blue numbers at the right and bottom tells the total number of instances for the current row or column. 
<br />For example the red number at the end of (right) the first row (980) tells you the total number of actual zeros in the dataset used.
<br />And the red number at the end of (bottom) the first column (983) tells you the number of times the model predicted the instance to be a zero.

<br />The accuracy and the confusion matrix (if for some reason does not open like above) are printed in the terminal.
<br />![alt text](https://github.com/doslindos/ml_stuff/blob/master/sources/example_images/confusion_terminal.png?raw=true)

### Autoencoder
TODO



# Mysql and Spotify API setup

This guide is for you who want to use MySQL data as a datset or Spotify API<br />

If you are using mysql functions make sure that your project has a **credentials.py** defined.
<br />Create it using credentials_example.py as a model.
<br />1. Move to where you did install the project.<br />`cd <project dir>`
<br />2. Copy credentials template.<br />`cp credentials_example.py credentials.py`

## Spotify credentials
Spotify credentials can be created at https://developer.spotify.com/dashboard/ 
1. It requires a Spotify account, but if you have one just sign-in.
2. After sign-in, go to **Dashboard** and click **Create an app**, fill info and click **Create**.
3. The **Dashboard** should now display your app. Click your app and you can find **CLIENT_ID** and **CLIENT_SECRET**.
4. Put **CLIENT_ID** and **CLIENT_SECRET** hash (weird string of numbers and letters) to your **credentials.py** file, inside the empty strings:
'client_id' and 'secret_key'.
<br />Done.

## MySQL credentials
Fill in the MySQL_connector_params. <br />More information: https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html 
<br />Connector params are given to the connector, therefore if you want to add attributes just add them to params with **key** as attribute name and **value** as input.


# Commands

# Command Info
## Fetch a dataset
Dataset is created with `<command>` **dataset**.<br />
#### Argument
| Argument | Flag | Info |
|-----------|------|---------|
<br />TODO

## Train a model
Training is called with `<command>` **train**.<br />
#### Argument
| Argument | Flag | Info |
|-----------|------|---------|
<br />TODO

## Test a model
Test function are called with `<command>` **test**.<br />
#### Argument
| Argument | Flag | Info |
|-----------|------|---------|
<br />TODO

