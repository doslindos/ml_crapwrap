from pathlib import Path
import unittest
from shutil import rmtree
from utils.utils import list_subfolder_in_folder, list_files_in_folder
from data import DatasetHandler

def loop_sub_handlers(handlers):
    for h_name, handler_values in handlers.items():
        for sub_h in handlers[h_name]['sub_handlers']:
            yield (h_name, sub_h)

class DatasetHandlerTester(unittest.TestCase):
    
    def setUp(self):
        #print("Called HandlerTester setup")
        #self.handlers = {}
        pass

    def available_handlers(self, handlers=None):
        handlers_path = Path("data", "handlers")
        self.handlers = {}
        if handlers is None:
            self.handler_names = [i.name for i in list_subfolder_in_folder(handlers_path) if '__' not in i.name]
        else:
            self.handler_names = handlers

        for h_name in self.handler_names:
            res_path =  handlers_path.joinpath(h_name, "resources")
            if res_path.exists():
                sources = [i for i in list_files_in_folder(res_path, '*.*', full=True)]
                self.handlers[h_name] = {'resources':sources}
            else:
                self.handlers[h_name] = {'resources':[None]}


    def create_handlers(self):
        # Create the handlers
        for handler_name in self.handler_names:
            for source in self.handlers[handler_name]['resources']:
                if isinstance(source, str):
                    ds_name = 'test_'+handler_name+'_'+source.split(".")[0]
                elif isinstance(source, type(None)):
                    ds_name = 'test_'+handler_name

                handler = DatasetHandler(handler_name, ds_name, source)
                if 'sub_handlers' not in self.handlers[handler_name].keys():
                    self.handlers[handler_name]['sub_handlers'] = [(handler_name, handler)]
                else:
                    self.handlers[handler_name]['sub_handlers'].append((handler_name, handler))

        print("Handlers initialized...")

    def load_function(self, load_input=None):
        # Load the data
        for handler_name, sub_handler in loop_sub_handlers(self.handlers):
            sub_name = sub_handler[0]
            handler = sub_handler[1]
            try:
                handler.load(load_input)
                print("Load works for ", sub_name)
            except Exception as e:
                print("Handler ", sub_name, " load error: ", e)

    def fetch_function(self, sample=None):
        # Fetch the data
        for handler_name, sub_handler in loop_sub_handlers(self.handlers):
            sub_name = sub_handler[0]
            handler = sub_handler[1]
            dataset = handler.fetch_preprocessed_data(sample)
            print("Fetching works for ", sub_name)
            if sample is None:
                self.assertIsInstance(dataset, tuple)
                self.assertEqual(len(dataset), 3)
                for ds in dataset:
                    self.assertIsInstance(ds, tfdata.Dataset)


    def destroy_test_datasets(self):
        ds_parents = []
        ds_files = []
        for handler_name, sub_handler in loop_sub_handlers(self.handlers):
            sub_name = sub_handler[0]
            handler = sub_handler[1]
            # Get the path to sub handlers test dataset
            path_to_test_ds = handler.data_fetcher.save_path
            if path_to_test_ds.is_dir():
                # Removes recursively the defined folder
                rmtree(path_to_test_ds)
            else:
                ds_file = path_to_test_ds.name
                parent = path_to_test_ds.parent
                parent_files = list_files_in_folder(parent, '*.*', full=True)
                if len(parent_files) == 1:
                    # Parent folder contains only one file
                    if parent_files[0] == ds_file:
                        # Removes recursively the defined folder
                        rmtree(parent)
                    else:
                        print("Save file doesn't contain ds file !? ", ds_file)
                        print("This should not happen")
                        exit()
                else:
                    # Collect path to ds and ds filenames for further processing
                    ds_files.append(ds_file)
                    ds_parents.append(parent)
        
        # Go through all dataset parent folder files to check that only test ds files are deleted
        if len(ds_parents) > 0:
            for parent in list(set(ds_parents)):
                parent_files = list_files_in_folder(parent, '*.*', full=True)
                if all(parent_files in ds_files):
                    # All files are created for the tests so delete the whole folder recursively
                    rmtree(parent)
                else:
                    # Unexpected file found in parent folder
                    # Loop through files in parent folder and delete only the ones found in ds_files
                    for p_file in parent_files:
                        if p_file in ds_files:
                            # Probably have to use parent.joinpath(p_file).unlink()
                            p_file.unlink()

