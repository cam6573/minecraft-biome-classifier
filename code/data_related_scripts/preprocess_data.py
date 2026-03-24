import pandas as pd
import numpy as np
import os
from glob import glob 
import random
import cv2 
import matplotlib.pylab as plt
import shutil
import math

random.seed(1000)

'''
biome 1 planes
biome 2 desert
biome 3 moutains
biome 6 swamp
biome 29 dark forest
biome 35 savanna 
'''


biomes = [(1,"plains"),
              (2, "desert"),
              (3, "mountains"),
              (6, "swamp"),
              (29, "dark_forest"), 
              (35, "savanna")]
    

PREPROCCESSED_DIR_PATH = "data/preprocessed/"
RAW_DATA_DIR_PATH = "data/raw/preprocessed_data/biome_"
NUMBER_OF_SAMPLES_PER_BIOME = 600

def get_image_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def is_trash(img_path):
    try:        
        img = cv2.imread(img_path)
    except:
        print(f"could not open image: {img_path}")
    if img is None:
        return None

    color_score = img.std()
    entropy = get_image_entropy(img)

    if color_score < 25 or entropy < 5.0:
        return True
    return False



def get_all_data():
    data = []

    for biome_id, biome_name in biomes:
        files = glob(RAW_DATA_DIR_PATH + str(biome_id) + "/*.jpg")
        random.shuffle(files)

        valid_files = []
        for f in files:
            if not is_trash(f):
                valid_files.append(f)

        selected_files = valid_files[:600]

        for f in selected_files:
            data.append((biome_name, f))

    return data
    
def split_up_data(data: list):
    random.shuffle(data)
    print(f"Data Set Size: {(len(data))}")

    training_data_set_size = math.ceil(len(data) * 0.70)
    validation_data_set_size = math.ceil(len(data)* 0.15)
    test_data_set_size = len(data) - training_data_set_size - validation_data_set_size

    print(f"Training data Set Size: {training_data_set_size}")
    print(f"Validation data Set Size: {validation_data_set_size}")
    print(f"Test data Set Size: {test_data_set_size}")

    
    training_data_set = []
    validation_data_set = []
    test_data_set = []

    for i in range(len(data)):
        if i <= training_data_set_size:
            training_data_set.append(data[i])
        elif i <= validation_data_set_size+ training_data_set_size:
            validation_data_set.append(data[i])
        else:
            test_data_set.append(data[i])

    print(f"Actual training data Set Size: {len(training_data_set)}")
    print(f"Validation data Set Size: {len(validation_data_set)}")
    print(f"Test data Set Size: {len(test_data_set)}")


    return training_data_set, validation_data_set, test_data_set


def save_files_to(folder_name:str, data:list):  
    folder_path = os.path.join(PREPROCCESSED_DIR_PATH,folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    index = 1
    for f in data:
        biome_name = f[0]
        file = f[1]

        file_path = os.path.join(folder_path,(f"{index}_{biome_name}.jpg"))
        shutil.copyfile(file, file_path)     
        index += 1
   


def main():
        data = get_all_data()

        if data is None:
            print("Issue getting the data")
            return
        else: 
            training_data_set, validation_data_set, test_data_set = split_up_data(data=data)
            save_files_to("training", training_data_set)
            save_files_to("validation", validation_data_set)
            save_files_to("test", test_data_set)

       



if __name__ == "__main__":
    main()


