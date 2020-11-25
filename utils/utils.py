from . import Path, nparray, npappend
from tensorflow import squeeze as tfsqueeze, stack as tfstack

from numpy import expand_dims as expdims

#TODO write comments

def list_files_in_folder(folder_path, suffix='*.py'):
    files = Path(folder_path).rglob(suffix)
    return [f.name.split('.')[0] for f in files]

def list_subfolder_in_folder(folder_path):
    content = folder_path.glob('*/')
    return [c for c in content if c.is_dir()]

def results_to_nplist(results):
    if isinstance(results, dict):
        labels = []
        initial = True
        for i, (label, result) in enumerate(results.items()):
            #print(label, result.shape)
            
            for data_instance in result:
                data_instance = expdims(data_instance, axis=0)
                if initial:
                    data = data_instance
                    initial = False
                else:
                    data = npappend(data, data_instance, axis=0)
                labels.append(label)

        #print(data.shape)
        return (data, labels)
    else:
        print("Results is not a dict...")

def duplicate_along(data, along_ax, label=None):
    data = tfsqueeze(tfstack([data for i in range(3)], axis=along_ax))
    return (data, label)

def input_check(x, allowed_input_types, checker):
    # Checks if input is in allowed types
    # In:
    #   x:                                      input for a function
    #   allowed_input_types:                    list, of all allowed input types
    #   checker:                                str, for errors

    for allowed in allowed_input_types:
        if isinstance(x, allowed):
            return

    print("Your input was not allowed type. Your input type = ", type(x))
    print("Variable was: ", checker)
    exit()

def path_check(path):
    # Checks if path exists
    # In:
    #   path:                                   Path object
    
    if path.exists():
        return True
    return False
