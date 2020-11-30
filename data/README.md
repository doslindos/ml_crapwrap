# Getting the data
Every time data is needed to perform a function in the application, a Dataset object is called (data/dataset.py) which will have functions and methods for different kinds of data fetching cases.
## Dataset object
The Dataset object calls handlers to actually perform the fetching of the data.
<br />Description of what a handler should have and do is defined in Data Handlers part.

### Inputs
The dataset object takes a name of the dataset as input.<br />It uses this name to call out the handler functions, so there should be a handler (defined in Data Handlers) created with this name.

### fetch_raw_data function
This function fetches the original data without preprocessing.
<br />Inputs for this function are sample size to be fetched and it returns a tensorflow dataset object.
### fetch_preprocessed_data
This function fetches the data and feeds it to the DataPreprocessor
<br />Inputs for this function is sample size to be fetched and it returns a tensorflow dataset object.

# Data Handlers

## Description
Handler is called from commandline arguments by its name, which is the folder name in here (data/handlers).
<br />Use the same names for files classes and functions defined here so that the commandline functions can work with the handlers.
<br />Inside a handler folder should be a fetch.py and preprocess.py file.

## fetch.py
<br />When creating a dataset, the Dataset object is called (data/dataset.py), which loads fetch.py module with the name given in handler argument (uses dataset name if handler argument is not given) and calls a class in the fetch.py called DatasetFetcher.
<br />DataFetcher is initialized with dataset name and source.<br/>Dataset name is used to name the created dataset and source is for implementing a source file into the loading functionality of data.
<br />After initialization a function/method of DatasetFetch named fetch_data is called, which is supposed to have everything needed to make the data usable by setup function (for example downloading it from a site and storing it in data/<handler name>/datasets/ folder).
<br />The function creates a tensorflow dataset object from the data and returns it.
<br />fetch_raw is called by Dataset object or DataPreprocesser to use the data.

## preprocess.py
<br />Dataset object initializes also DataPreprocessor when it is initialized.
<br />DataPreprocessor should have a function called preprocess, which takes in a dataset, preprocesses the dataset and returns preprocessed data in a Tensorflow dataset object.
<br />The DataPreprocessor should be designed so that you can pass in from one instance to many instances, because it is called from different places in the application.
<br />For example when training the preprocess function is called when a set of data has to be fed to the model and when testing the model there might be a case when just one input is converted to be fed to the model.
<br />This is ensured by creating a seperate preprocess function which is used for every instance of data.
<br />Every function that uses the preprocess will loop the tensorflow dataset object.
