from .. import Path, run_function, fetch_resource, dataset_generator, npargmax, open_fileGUI, npappend
from cmd import Cmd
from .util import get_dataset
from csv import writer as csvwriter
from tensorflow.data import Dataset as tfdataset

def write_result(path, out, with_ids):

    print(out.shape)
    if hasattr(out, 'numpy'):
        out = out.numpy()
    
    with path.open('w', encoding='utf-8') as f:
        writer = csvwriter(f, delimiter=",", lineterminator='\n')
        for i, pred in enumerate(out):
            
            # Ask for headers
            if i == 0:
                h = input("Do you want headers (y/n)?")
                if h == 'y':
                    headers = input("Give headers (comma seperated):")
                    headers = headers.split(',')
                    writer.writerow(headers)

            if isinstance(pred, str) or not hasattr(pred, '__iter__'):
                result = pred
            else:
                result = npargmax(pred)
            if with_ids:
                result = [i+1, result]
            else:
                result = [result]

            writer.writerow(result)

class ModelTesterCLI:

    def __init__(self, model):
        self.model = model
        prompt = self.ModelPrompt()
        prompt.set_model(model)
        prompt.cmdloop()

    class ModelPrompt(Cmd):

        def set_model(self, model):
            # Initializes the model to be used

            self.model_meta = { "name": model.conf_name, "Configurations": model.c }
            self.model = model

        def choose_data(self, use):
            # Choose data to use

            datas = [ d for d in ['train', 'test', 'validate', 'selected_data'] if hasattr(self, d)]
            
            if datas:
                if use not in datas:
                    while use not in datas:
                        use = input("Please choose one of these "+str(datas)+ ": ")

                return getattr(self, use)

            else:
                print("Please get data or dataset with 'get_dataset' or 'get_data' first...")

        def do_model_info(self, inp):
            # Prints info of the mode
            
            print(self.model_meta)

        def do_get_dataset(self, _ = None):
            # Select a loaded dataset

            self.train, self.validate, self.test = get_dataset()

        def do_get_data(self, rootpath = None):
            # Select custom set of data

            if rootpath is None:
                rootpath = Path.cwd()

            path = open_fileGUI(rootpath)
            self.ds = tfdataset.from_tensor_slices(fetch_resource(path)[0])

        def do_show_data(self, use = None):
            # Display data
            data = self.choose_data(use)
            #TODO

        def do_prediction_file(self, filename = 'results.csv', with_ids = True):
            
            if filename == '':
                filename = 'results.csv'

            path = Path(Path.cwd(), "sources", "kaggle")
            if not path.exists():
                path.mkdir()

            path = path.joinpath(filename)

            if not hasattr(self, 'ds'):
                self.ds = self.test

            for i, batch in enumerate(self.ds.batch(100)):
                if len(batch) == 2:
                    batch = batch[0]
                out = self.model.run(batch).numpy()
                if i == 0:
                    all_outs = out
                else:
                    all_outs = npappend(all_outs, out, axis=0)
            print(all_outs.shape)
            write_result(path, all_outs, with_ids)

            
        def do_exit(self, inp):
            print("Goodbye!")
            return True

