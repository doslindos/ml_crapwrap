from .. import Path, Tk, ttk, filedialog, StringVar, build_blueprint, open_fileGUI, run_function, random_sample, get_dataset_info, show_data_tk, tfreshape, exit, dataset_generator, npargmax, nparray, npreshape, npsqueeze, tfdata, is_tensor, fetch_resource, take_image_screen, get_function_attr_values
from .GUI_config import conf
from utils.modules import fetch_model
from .util import get_dataset

class ModelTesterGUI:

    def __init__(self, model):
        self.root = Tk()
        self.build_start_window()
        
        self.ds = StringVar()
        self.ds.set("Training")
        if isinstance(model, tuple):
            self.model_name = model[0]
            self.conf_name = model[1]
        else:
            self.model = model

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.mainloop()

    def quit(self):
        exit()

    def build_start_window(self):
        # Build the tkinter window
        self.widgets = build_blueprint(self, conf)

    def select_file(self):
        # Fetch the file path
        path = open_fileGUI(Path.cwd().joinpath("sources"))
        
        # Shortcut for testing
        #path = Path('C:\\', 'Users', '35850', 'Projects', 'PYTHON', 'oppari', 'sources', 'example_images', 'confusion_m_example.png')
        #print(path)
        
        # Setup model if not done allready
        if not hasattr(self, 'model'):
            self.model = fetch_model(self.model_name, self.conf_name)

        
        # Preprocess data

        # Set up the data to display in the GUI
        self.setup_data(dataset=False)

    def dataset_change(self):
        if self.ds.get() == "Training":
            if not hasattr(self, 'train_ds_gen'):
                self.train_ds_gen = dataset_generator(self.train, 1)
            self.dataset_generator = self.train_ds_gen
        elif self.ds.get() == "Validation":
            if not hasattr(self, 'val_ds_gen'):
                self.val_ds_gen = dataset_generator(self.validate, 1)
            self.dataset_generator = self.val_ds_gen
        elif self.ds.get() == "Testing":
            if not hasattr(self, 'test_ds_gen'):
                self.test_ds_gen = dataset_generator(self.test, 1)
            self.dataset_generator = self.test_ds_gen
        
        self.setup_data(dataset=True)
    
    def select_dataset(self, change=False):
        self.train, self.validate, self.test = get_dataset()
    
        self.dataset_change()    

        
        if not hasattr(self, 'model'):
            self.model = fetch_model(self.model_name, self.conf_name)
        #print(self.model)
        #print(self.model.configurations)
        
        # Random order of for fetching samples
        #self.data_order = random_sample(range(self.ds_length), self.ds_length)
        
        # Set up the dataset to display in the GUI

    def setup_data(self, dataset=True):
        if not hasattr(self, 'data_frame'):
            self.data_frame = ttk.Frame(self.root)
            self.data_frame.grid(row=1, column=0)
            ttk.Label(self.data_frame, text="Feed data to model:").grid(row=0, column=1)
            ttk.Button(self.data_frame, text="Classify", command=lambda:self.feed_to_model(True)).grid(row=1, column=1)
            ttk.Button(self.data_frame, text="Raw output", command=lambda:self.feed_to_model(False)).grid(row=2, column=1)
        
            if dataset:
                self.take_next_instance()
            
                ttk.Label(self.data_frame, text="Select dataset to use:").grid(row=0, column=0)
                ds_options = ['', 'Training', 'Validation', 'Testing']
                ttk.OptionMenu(self.data_frame, self.ds, *ds_options).grid(row=1, column=0)
                self.ds.trace('w', lambda n, i, m: self.dataset_change())
                ttk.Button(self.data_frame, text="Next", command=lambda:self.take_next_instance()).grid(row=2, column=0)
        
        else:
            self.show_instance()

    def take_next_instance(self):
        self.data, self.label = next(self.dataset_generator)
        self.show_instance()
    
    def show_instance(self, output=False):
        if not output:
            # Create input images
            data = npsqueeze(self.data)
            if len(data.shape) == 1:
                data = data.reshape(1, data.shape[0])
            if not hasattr(self, 'input_figure'):
                # Create models input
                self.model_input_figure = show_data_tk(
                        self.data_frame, 
                        data, 
                        self.label, 
                        pack={'row':4, 'column':0}
                        )

            # Update input images
            else:
               # Update models input
                self.model_input_figure = show_data_tk(
                        self.model_input_figure, 
                        data, 
                        self.label
                        )
    
        else:
            if len(self.output.shape) > 3:
                self.output = npsqueeze(self.output)

            if not hasattr(self, 'output_figure'):
                # Create output figure
                self.output_figure = show_data_tk(
                        self.data_frame, 
                        self.output, 
                        self.label, 
                        pack={'row':4, 'column':1}
                        )
            else:
                # Update output figure
                self.output_figure = show_data_tk(
                        self.output_figure, 
                        self.output, 
                        self.label
                        )

    def feed_to_model(self, classify=True):
        output = self.model.run(self.data, False)
        label = self.label
        if hasattr(output, 'numpy'):
            output = output.numpy()
        if hasattr(label, 'numpy'):
            label = label.numpy()
        
        if classify:
            print("Prediction:", npargmax(output), " True label: ", label)
        else:
            self.output = output
            self.show_instance(True)
