from subprocess import call as sub_call
from .. import Path, input_check
from tensorflow.data.experimental import save as tfsave, load as tfload
from utils.utils import list_subfolder_in_folder
from pickle import dump as pkldump, load as pklload

def kaggle_submit(competition, filepath, message=""):
    # In:
    #   competition:                    str, Name of the competition
    #   filepath:                       str, path where to get results
    #   message:                        str, message to submit

    input_check(competition, [str], "competition_name from kaggle_download")
    input_check(filepath, [Path], "path from kaggle_download ")
    input_check(f, [str], "ds_name from kaggle_download")
    
    if filepath.exists():
        # Build call
        call = ["kaggle", "competitions", "submit", "-f", path, "-m", message, competition_name]
        # Make download call
        print(sub_call(call))

def save_tfdataset(path, datasets):
    # Saves datasets
    # In:
    #   path:                           Path, to save folder
    #   datasets:                       tuple, datasets train, validate and test
    
    for key, ds in datasets.items():
        tfsave(ds, str(path.joinpath(key)))
        with path.joinpath(key, "spec.pkl").open('wb') as fs:
            pkldump(ds.element_spec, fs)

def load_tfdataset(path, dtype="float32"):
    # Reads datasets
    # In:
    #   path:                           Path, to records
    #   dtype:                          str, datatype
    # Out:
    #   tuple:                          datasets
    
    datasets = {}
    for f in list_subfolder_in_folder(path):
        with path.joinpath(f ,"spec.pkl").open('rb') as fl:
            spec = pklload(fl)
        
        ds = tfload(
                str(path.joinpath(f)),
                spec
                )

        datasets[f.name] = ds
        
    return datasets

def save_encoders(path, encoders):
    if path.name != "encoders.pkl":
        path = path.joinpath("encoders.pkl")
    with path.open('wb') as fs:
        pkldump(encoders, fs)

def load_encoders(path):
    if path.name != "encoders.pkl":
        path = path.joinpath("encoders.pkl")
    with path.open('rb') as fl:
        return pklload(fl)

