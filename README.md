
## Project : Minecraft biome classfier

### Abstract


### Developers
- Chelsea Malach
- Carla Lopez
- Sara Zangrilli
- Sebastian Canakis


### How to run project 

Set venv
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 
python -m pip install kagglehub pandas numpy matplotlib opencv-python scikit-image
```

Download DataSet and Select 600 random images for each biom:
```
python code/data_related_scripts/download_data.py
python code/data_related_scripts/preprocess_data.py 
```

## Train Random Forest Model for Image Classification

To train the random forest model, do the following

1. Run script to create feature and label matrices ( extract HOG and color features): 
```
python code/random_forest/feature_extraction/extract_features.py
```

2. Perform hyperparameter tuning to find best parameters 
```
python code/random_forest/tuning.py
```
3. Train Random Forest Model WITH best hyperparameters: 
```
python code/random_forest/training.py
```


#### Project Structure:
```
.
├── README.md
├── code
│   └── data_related_scripts
│   |    ├── download_data.py
│   |    └── preprocess_data.py
|   └── random_forest
│   |    ├── feature_extraction
|   |    |      ├──extract_features.py  
|   |    |   
│   |    └── training.py
│   | 
|   |
└── data
    ├── preprocessed
    │   ├── dark_forest
    │   ├── desert
    │   ├── mountains
    │   ├── plains
    │   ├── savanna
    │   └── swamp
    ├── processed
    │   └── HOG
    │       ├── HOG_vectors
    │       │   ├── dark_forest
    │       │   ├── desert
    │       │   ├── mountains
    │       │   ├── plains
    │       │   ├── savanna
    │       │   └── swamp
    │       └── images
    │           ├── dark_forest
    │           ├── desert
    │           ├── mountains
    │           ├── plains
    │           ├── savanna
    │           └── swamp
    └── raw
        └── preprocessed_data

```