
## Project : Minecraft biome classfier

### Abstract


### Developers
- Chelsea Malach
- Carla Lopez
- Sara Zangrilli
- Sebastian Canakis


### How to run project 

Set venv
Mac
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 
```

2. Install all required dependencies/libraries
```
python -m pip install -r requirements.txt
```

Windows:
```
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install kagglehub pandas numpy matplotlib opencv-python scikit-image
```

If you run into issues run the following:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```


Download DataSet and Select 600 random images for each biome:
```
python code/data_related_scripts/download_data.py
python code/data_related_scripts/preprocess_data.py 
```

3. Tran and Evaluate SVM, Random Forest or CNN for Minecraft Biome Classification


## Train Random Forest Model for Image Classification
After downloading the dataset and preprocessing the images, execute the following steps:

## Full Pipeline Execution

To run the entire end-to-end process from raw pixels to final evaluation of the Rain Forest Model run the main script from the root directory:  
```
python code/random_forest/main.py
```
What this does:

1. Feature Extraction: Converts screenshots into numerical feature matrices.

2. Hyperparameter Tuning: Runs a Grid Search to find the best model settings.

3. Final Training: Trains the model using the best parameters.

4. Evaluation: Generates and saves performance graphs and tables

## Running each component individually

To run each part of the Random Forest pipeline independently:

1. Feature Extraction: Extracts feature (HOG,color histograms) matrix and label vector into .npy files for faster training. 
```
python code/random_forest/feature_extraction/extract_features.py
```

2. Hyperparameter Tuning: Finds the optimal settings for the Random Forest and saves them to best_params.json

```
python code/random_forest/training/tuning.py
```

3. Train Random Forest Model WITH best hyperparameters: Trains the final model, and performs model evaluation (accuracy table and confusion matrix)

```
python code/random_forest/training/training.py
```



#### CNN:

The CNN model should be already trained and under `./model/CNN/minecraft_biome_cnn.keras`


#### Project Structure:
```
.
├── README.md
├── code
│   ├── data_related_scripts
│   │   ├── download_data.py
│   │   └── preprocess_data.py
│   └── random_forest
│       ├── main.py
│       ├── evaluation
│       ├── feature_extraction
│       │   └── extract_features.py
│       └── training
│           ├── load_data.py
│           ├── training.py
│           └── tuning.py
└── data
    ├── preprocessed
    │   ├── dark_forest
    │   ├── desert
    │   ├── mountains
    │   ├── plains
    │   ├── savanna
    │   └── swamp
    └── raw
        └── preprocessed_data

```