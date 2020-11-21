from . import Path, nparray
import cv2
from csv import reader as csv_reader

def take_image_screen(size=[]):
    pass

def fetch_resource(path, desired_shape=None, desired_dtype=None):
    # Handle resource fetching
    # In:
    #   path:                       Path Object
    #   desired_shape:              ?
    #   desired_dtype:              ?

    if path.exists():
        suf = path.suffix
        
        # Handle unspecfied data type
        if desired_dtype is not None:
            dtype = desired_dtype
        else:
            dtype = 'float32'

        if suf in ['.jpg', '.png']:
            if desired_shape is None:
                # No desired input shape defined
                # Return image as numpy array, with RGB color type
                return nparray(cv2.imread(path.as_posix(), cv2.COLOR_BGR2RGB), dtype=dtype)
            else:
                # Define shapes
                color_dim = desired_shape[-1]
                h = desired_shape[0]
                w = desired_shape[1]
                
                # Handle different color types
                if color_dim == 1:
                    col = cv2.COLOR_BGR2GRAY
                elif color_dim == 3:
                    col = cv2.COLOR_BGR2RGB

                # Read image
                img = cv2.cvtColor(cv2.imread(path.as_posix()), col)
                # Resize image for the model
                resized = cv2.resize(img, (h, w))
                
                # Reshape it for the model (add batch dimension)
                resized = resized.reshape((1, h, w, color_dim))
                # Convert to a specific data type
                resized = resized.astype(dtype)
                
                return (img, resized)

        elif suf in ['.csv']:
            
            def error(value):
                print("Input must be numeric...  your input ", value ," can't be converted to a float...")
                exit()
            
            # Desired input array length
            array_len = desired_shape[-1]

            # Read csv file
            csv_file = csv_reader(path.open('r', encoding='utf8'), delimiter=';')
            # Loop through rows
            n_rows = []
            for row_num, row in enumerate(csv_file):
                # Loop through instances in a row
                for i, value in enumerate(row):
                    # Convert instance to float and replace the old value with converted
                    if isinstance(value, str) or isinstance(value, int):
                        try:
                            value = float(value)
                        except ValueError:
                            #TODO print error
                            error(value)
                    else:
                        #TODO print error
                        error(value)

                    row[i] = value
                
                n_rows.append(row)

            n_rows = nparray(n_rows, dtype=dtype)
            
            if desired_shape is not None:
                if len(n_rows) > 1:
                    n_rows = n_rows.reshape((1, len(n_rows), array_len))
                else:
                    n_rows = n_rows[0].reshape((1, array_len))


            return (n_rows, n_rows)


