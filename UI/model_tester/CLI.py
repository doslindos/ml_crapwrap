from .. import Path, run_function
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
            self.model_meta = { "name": model.conf_name, "Configurations": model.c }
            self.model = model

        def do_set_dataset(self, inp):
            self.train, self.test, self.validate = get_dataset()

        def do_exit(self, inp):
            print("Goodbye!")
            return True

        def do_model_info(self, inp):
            print(self.model_meta)
