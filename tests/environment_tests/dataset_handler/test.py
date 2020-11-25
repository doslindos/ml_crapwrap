import unittest
from utils.utils import list_subfolder_in_folder
from pathlib import Path
import unittest


class DatasetHandlerTester(unittest.TestCase):
    def setUp(self):
        print("Setup")
        self.folder_names = [i.name for i in list_subfolder_in_folder(Path("data", "handlers")) if '__' not in i.name]
        print(self.folder_names)
    def test_fun(self):
        print("Carmageddon")

class dormund_create(DatasetHandlerTester):
    def setUp(self):
        print("Crearte")
    
    def test_create_m(self):
        print("Hoplaa")
        
