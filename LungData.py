### IMPORTS ###
import os
import glob
from PIL import Image
from torch.utils.data import Dataset

# Define custom class for Lung dataset
class Lung(Dataset):
    
    def __init__(self, transform=None):  # lung_type: str
        
        # Define path to Lung image set folder
        self.imgs_path = "lung_image_sets/"
        
        # Create a path for each class in Lung image set folder
        file_list = glob.glob(self.imgs_path + "*") 
        
        self.data = []
        for class_path in file_list:
            # Get class name
            class_name = class_path.split('/')[-1]
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])
        
        # Dictionary for the different classifications
        self.class_map = {"lung_n": 0, "lung_scc": 1, "lung_aca": 2}
        self.transform = transform

    def __len__(self):
        # To return the length of the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # To retrieve an image and its class id
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id
