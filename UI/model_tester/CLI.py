from cmd import Cmd

class ModelTesterCLI:

    def __init__(self, model):
        self.model = model
        prompt = self.ModelPrompt()
        prompt.set_model(model)
        prompt.cmdloop()

    class ModelPrompt(Cmd):
        
        def set_model(self, model):
            self.model = model

        def do_exit(self, inp):
            print("Hep")
            return True

        def do_model(self, filepath):
            print(self.model)
            print(path)
