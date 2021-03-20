from .. import Path, run_function, fetch_resource
from cmd import Cmd
from .util import get_dataset

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

        def do_get_dataset(self, _):
            # Select a loaded dataset

            self.train, self.test, self.validate = get_dataset()

        def do_get_data(self, rootpath = None):
            # Select custom set of data

            if rootpath is None:
                rootpath = Path.cwd()

            path = open_fileGUI(rootpath)
            self.selected_data = fetch_resource(path)

        def do_show_data(self, use = None):
            # Display data
            self.data = self.choose_data(use)

            
        def do_exit(self, inp):
            print("Goodbye!")
            return True

