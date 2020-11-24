# Machine learning function wrapper

## Description
The purpose of this project has been for me to practice Python and Machine Learning
<br />It is a wrapper for trying out Machine Learning functions. 
The project came to be as I tried to learn how to use several Machine Learning models and was tired to write a new script for every single one of them.<br />
After several scripts lying all ower my computer I decided to create a single repo for them<br />
If you want to test out the project it will require from you some basic knowledge of using the commandline but the instructions are hopefully clear enough for a quick try.<br />
<br />This is mainly just a place for me to store my code and the instructions for me not to forget how this works, so it is not stable, everything that is in the guide is tested on Windows and Mac (not the latest cahnges).

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
4. Move to the created project<br />`cd ml_framework`
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
First go to the project directory in commandline (directory where you downloaded the project in the setup (name is ml_framework))<br />`cd <project directory>`
<br />Next download the MNIST dataset:<br />`python main.py dataset -d mnist -ff tfds_fetch`
<br />After it is done, you can train your first MNIST model:<br />`python main.py train -d mnist -c mnist_dense`
<br />This command will train a dense classifier neural network with default settings (can be found and modified NeuralNetworks/configurations.py).
<br />The training will take a while, but after it is done and if everything did go well, in the commandline should be text "Training finished..."<br />
Now you can try out your model! <br />Type:<br />`python main.py test -d mnist -test classification_test`<br />
A folder selection window should pop up. (Mac example of the file selection)
![alt text](https://github.com/doslindos/ml_framework/blob/master/sources/example_images/mac_select_model.png?raw=true) Choose the model you trained by double clicking the first folder on the popup window and select the model **folder** **(Trained models folders are named with time stamp from the time the training occured)**. Click on the model you want to use and select Choose.
<br />Condusion Matrix should be displayed on the screen.
![alt text](https://github.com/doslindos/ml_framework/blob/master/sources/example_images/confusion_m_example.png?raw=true)
<br />Confusion Matrix displays the predictions of a classification model. At the top are the models predictions and on the left the real labels of the inputs.<br />
Prediction is right when the top and the right numbers match. So for example the most upper left corner signifies predicted 0 images which actually are a 0 image.

<br />Some working default training commands:
<br />MNIST:
<br />Train a dense mnist image classifier:
<br />`python create.py train -d mnist -c mnist_dense`
<br />Train a convolutional-dense mnist image classifier:
<br />`python create.py train -d mnist -c mnist_conv`
<br />Train a dense mnist image autoencoder:
<br />`python create.py train -d mnist -c mnist_basic_autoencoder`
<br />Train a convolutional-dense mnist image autoencoder:
<br />`python create.py train -d mnist -c mnist_conv_autoencoder`

#TODO
More visual demonstrations...

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


# Guide example
The `data/data_scripts/example.sql` file provides an template for a MySQL dataset fetch.
<br />1. Open command prompt or terminal and navigate to the folder where project was installed:<br />`cd <project directory>`
<br />2. Activate virtualenvironment:<br />**Mac and Linux**<br />`source env/bin/activate`<br />**Windows**<br />`env\Scripts\activate`
<br />3. Create the dataset:<br />`python main.py dataset -d <src.sql> -ff msyqldb_fetch -cf <create function name> -name <dataset name>`<br />
where <src.sql> is the sql file, for example: example.sql<br /><create function name> is the name of create function, for example: spotify<br /><dataset name> is the name for the created dataset.
TODO

# Commands

All commands are called from active virtualenv with `python main.py <command>`<br />
All command attributes can be displayed with -h flag for example: `python main.py <command> -h`

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

