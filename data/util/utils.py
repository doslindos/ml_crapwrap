from subprocess import call as sub_call
from .. import Path, input_check

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
