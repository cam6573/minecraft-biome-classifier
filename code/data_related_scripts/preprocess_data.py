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



def get_biomes(folder_num, biome_name):
    os.makedirs(PREPROCCESSED_DIR_PATH + str(biome_name))
    try: 
        biome = glob(RAW_DATA_DIR_PATH+ str(folder_num) + "/*.jpg")
    except:
        print("biomes could not be found in " + RAW_DATA_DIR_PATH)
        return
    
    already_collected_set = set()
    index = 0
    while index < NUMBER_OF_SAMPLES_PER_BIOME:
        current_biome_number = random.randrange(0,len(biome))
        if(already_collected_set.__contains__(current_biome_number)):
                continue
        else:
            source_path = biome[current_biome_number]
            if(not is_trash(img_path=source_path)):
                new_name = f"{biome_name}_{index}.jpg"
                destination_path = os.path.join(PREPROCCESSED_DIR_PATH, biome_name, new_name)
                already_collected_set.add(current_biome_number)
                index += 1

                shutil.copyfile(source_path, destination_path)


def get_all_data():
    # os.makedirs(os.path.join(PREPROCCESSED_DIR_PATH, "data"))
    try: 
        data  = []
        for biome in biomes:
            biome_id = biome[0]
            biome_name = biome[1]
            files = glob(RAW_DATA_DIR_PATH + str(biome_id) + "/*.jpg")
            random.shuffle(files)
            for i in range(600):
                data.append((biome_name, files[i]))
        return data
    except:
        print("biomes could not be found in " + RAW_DATA_DIR_PATH)
        return None
    
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

def main():
        training_data_set, validation_data_set, test_data_set = split_up_data(get_all_data())

        os.makedirs

if __name__ == "__main__":
    main()


