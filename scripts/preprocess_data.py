import pandas as pd
import numpy as np
import os
from glob import glob 
import random
import cv2 
import matplotlib.pylab as plt
import shutil




'''
biome 1 planes
biome 2 desert
biome 3 moutains
biome 6 swamp
biome 29 dark forest
biome 35 savanna 
'''

PROCCESSED_DIR_PATH = "data/processed/"
PREPROCCESSED_DIR_PATH = "data/raw/preprocessed_data/biome_"
NUMBER_OF_SAMPLES_PER_BIOME = 500

def get_biomes(folder_num, biome_name):
    os.makedirs(PROCCESSED_DIR_PATH + str(biome_name))
    try: 
        biome = glob(PREPROCCESSED_DIR_PATH+ str(folder_num) + "/*.jpg")
    except:
        print("biomes could not be found in " + PREPROCCESSED_DIR_PATH)
        return
    
    already_collected_set = set()
    index = 0
    while index < NUMBER_OF_SAMPLES_PER_BIOME:
        current_biome_number = random.randrange(0,len(biome))
        if(already_collected_set.__contains__(current_biome_number)):
                continue
        else:
            index += 1
            already_collected_set.add(current_biome_number)
            source_path = biome[current_biome_number]
            new_name = f"{biome_name}_{index}.jpg"
            destination_path = os.path.join(PROCCESSED_DIR_PATH, biome_name, new_name)

            shutil.copyfile(source_path, destination_path)



def main():
    biomes = [(1,"plains"),
              (2, "desert"),
              (3, "mountains"),
              (6, "swamp"),
              (29, "dark_forest"), 
              (35, "savanna")]
    
    for biome in biomes:
        get_biomes(folder_num=biome[0], biome_name=biome[1])



if __name__ == "__main__":
    main()


