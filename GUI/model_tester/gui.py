from .. import Path, Tk, ttk, filedialog, StringVar, build_blueprint, open_fileGUI, open_dirGUI, Preprocess, run_function, random_sample, get_dataset_info, fetch_model, show_data_tk, tfreshape, exit, dataset_generator, npargmax, nparray, npreshape, npsqueeze, tfdata, is_tensor, fetch_resource, take_image_screen, get_function_attr_values, normalize
from .GUI_config import conf

class Model_tester:

    def __init__(self, model, preprocess_function):
        self.root = Tk()
        self.build_start_window()
        
        self.use_test = True
        if isinstance(model, tuple):
            self.model_name = model[0]
            self.conf_name = model[1]
        else:
            self.model = model
        
        self.pf = preprocess_function

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

        # Fetch resource
        self.original_data, converted = fetch_resource(path, desired_shape=self.model.c['input_shape'])
        # Set labels
        self.original_label = "From file"
        self.processed_label = "From file"
        #print(self.original_data.shape, converted.shape)
        
        # Initialize preprocessing
        prep_pipe = Preprocess()
        #print(self.model.conf_name)
        
        # Fetch normalization function used
        func = get_function_attr_values(
                getattr(prep_pipe, self.pf), 
                'norm_function'
                )['norm_function']

        # Use preprocessing normalization function
        normed = run_function(normalize, func, {'data':converted})
        # Handle data out from returned norm function output
        if isinstance(normed, tuple):
            self.processed_data = normed[0]
        else:
            self.processed_data = normed
        
        # Set up the data to display in the GUI
        self.setup_data(dataset=False)

    def select_dataset(self):
        # Fetch the dataset path
        dataset = Path(open_dirGUI(Path("data", "created_datasets")))
        # Preprocess the dataset
        prep_pipe = Preprocess()
        run_function(prep_pipe, self.pf, {'dataset_name':dataset.name})
        # Save the datasets to Model_testser
        if self.use_test:
            i = 1
        else:
            i = 0
        
        # Take one dataset from preprocess function
        # Orignal data is used to show values and preprocessed is fed to the model
        self.original = prep_pipe.original_data
        if isinstance(prep_pipe.original_data, tuple):
            if hasattr(self.original[i], 'batch'):
                self.original = self.original[i]
            elif isinstance(self.original[i], dict):
                self.original = tfdata.Dataset.from_tensor_slices({'x':self.original[i]['x'], 'y':self.original[i]['y']})
        
        self.processed = prep_pipe.preprocessed_dataset
        if isinstance(prep_pipe.preprocessed_dataset, tuple):
            if hasattr(self.processed[i], 'batch'):
                self.processed = self.processed[i]
            elif isinstance(self.processed[i], dict):
                self.processed = tfdata.Dataset.from_tensor_slices({'x':self.processed[i]['x'], 'y':self.processed[i]['y']})
        
        ds_info = get_dataset_info(prep_pipe.preprocessed_dataset[i], True)
        # Use test dataset information which is the first index
        self.ds_length = ds_info[0]
        self.data_size = ds_info[1]
        self.data_type = ds_info[2]

        
        if not hasattr(self, 'model'):
            self.model = fetch_model(self.model_name, self.conf_name)
        #print(self.model)
        #print(self.model.configurations)
        
        # Random order of for fetching samples
        #self.data_order = random_sample(range(self.ds_length), self.ds_length)
        
        # Set up the dataset to display in the GUI
        self.setup_data(dataset=True)

    def setup_data(self, dataset=True):
        if not hasattr(self, 'data_frame'):
            self.data_frame = ttk.Frame(self.root)
            self.data_frame.grid(row=1, column=0)
            ttk.Button(self.data_frame, text="Classify", command=lambda:self.feed_to_model(True)).grid(row=1, column=0)
            ttk.Button(self.data_frame, text="Raw output", command=lambda:self.feed_to_model(False)).grid(row=2, column=0)
        
        if dataset:
            self.dataset = dataset_generator(self.original, 1)
            self.processed = dataset_generator(self.processed, 1)
        
            self.take_next_instance()
        
            ttk.Button(self.data_frame, text="Next", command=lambda:self.take_next_instance()).grid(row=0, column=0)
        
        else:
            self.show_instance()

    def take_next_instance(self):
        self.original_data, self.original_label = next(self.dataset)
        # Shape for displaying, (remove the batch, shape)
        #print(self.original_data, self.original_label)
        if is_tensor(self.original_data):
            self.original_data = tfreshape(self.original_data, [i for i in self.original_data.shape[1:]])
        # Handle numpy based dataset (for example spotify)
        elif isinstance(self.original_data, list):
            self.original_data = nparray(self.original_data)
            if len(self.original_data.shape) < 2:
                self.original_data = npreshape(self.original_data, (1, self.original_data.shape[0]))
        self.processed_data, self.processed_label = next(self.processed)
        self.show_instance()
    
    def show_instance(self, output=False):
        if not output:
            # Create input images
            processed = npsqueeze(self.processed_data)
            if len(processed.shape) == 1:
                processed = processed.reshape(1, processed.shape[0])
            if not hasattr(self, 'input_figure'):
                # Create original input
                self.input_figure = show_data_tk(
                        self.data_frame, 
                        self.original_data, 
                        self.original_label, 
                        pack={'row':3, 'column':0}
                        )
                # Create models input
                self.model_input_figure = show_data_tk(
                        self.data_frame, 
                        processed, 
                        self.processed_label, 
                        pack={'row':4, 'column':0}
                        )

            # Update input images
            else:
                # Update original input
                self.input_figure = show_data_tk(
                        self.input_figure, 
                        self.original_data, 
                        self.original_label
                        )
               # Update models input
                self.model_input_figure = show_data_tk(
                        self.model_input_figure, 
                        processed, 
                        self.processed_label
                        )
    
        else:
            if len(self.output.shape) > 3:
                self.output = npsqueeze(self.output)

            if not hasattr(self, 'output_figure'):
                # Create output figure
                self.output_figure = show_data_tk(
                        self.data_frame, 
                        self.output, 
                        self.original_label, 
                        pack={'row':3, 'column':1}
                        )
            else:
                # Update output figure
                self.output_figure = show_data_tk(
                        self.output_figure, 
                        self.output, 
                        self.original_label
                        )

    def feed_to_model(self, classify=True):
        output = self.model.run(self.processed_data, False)
        label = self.processed_label
        if hasattr(output, 'numpy'):
            output = output.numpy()
        if hasattr(label, 'numpy'):
            label = label.numpy()
        
        if classify:
            print("Prediction:", npargmax(output), " True label: ", label)
        else:
            self.output = output
            self.show_instance(True)
