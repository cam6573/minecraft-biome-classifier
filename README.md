
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

Download DataSet and Select 600 random images for each biome:
```
python code/data_related_scripts/download_data.py
python code/data_related_scripts/preprocess_data.py 
```


## Train AdaBoost Model for Image Classification - Developer: Chelsea Malach

After downloading the dataset and preprocessing the images, execute the following steps:

## Full Pipeline Execution

To run the entire end-to-end process from raw pixels to final evaluation of the AdaBoost Model run the main script from the root directory:  
```
python code/adaboost/main.py
```
What this does:

1. Feature Extraction: Converts Minecraft biome screenshots into numerical feature matrices using HOG and color histogram features.

2. Dataset Processing: Loads images from existing preprocessed training, validation, and test folders (found in data/preprocessed) and converts them into feature vectors.

3. Model Training: Trains an AdaBoost classifier using a decision stump as the weak learner.

4. Evaluation: Evaluates performance on training, validation, and test sets. Outputs accuracy, precision, recall, F1-score, and a full classification report.

## Running each component individually

To run each part of the pipeline independently:

1. Feature Extraction: Extracts HOG and color histogram features from pre-split image datasets and saves them as numerical feature matrices (.npy files) for faster training. 
```
python code/adaboost/extract_features.py
```

2. Train AdaBoost Model: Trains the AdaBoost model using the extracted feature matrices and performs model evaluation using accuracy metrics and a classification report.

```
python code/adaboost/train_adaboost.py
```



## Train Random Forest Model for Image Classification

After downloading the dataset and preprocessing the images, execute the following steps:

## Full Pipeline Execution

To run the entire end-to-end process from raw pixels to final evaluation of the Random Forest Model run the main script from the root directory:  
```
python code/random_forest/main.py
```
What this does:

1. Feature Extraction: Converts screenshots into numerical feature matrices.

2. Hyperparameter Tuning: Runs a Grid Search to find the best model settings.

3. Final Training: Trains the model using the best parameters.

4. Evaluation: Generates and saves performance graphs and tables

## Running each component individually

To run each part of the pipeline independently:

1. Feature Extraction: Extracts feature (HOG,color histograms) matrix and label vector into .npy files for faster training. 
```
python code/random_forest/feature_extraction/extract_features.py
```

2. Hyperparameter Tuning: Finds the optimal settings for the Random Forest and saves them to best_params.json

```
python code/random_forest/tuning.py
```

3. Train Random Forest Model WITH best hyperparameters: Trains the final model, and performs model evaluation (accuracy table and confusion matrix)

```
python code/random_forest/training.py
```

#### Project Structure:
```
.
├── README.md
├── code
│   ├── data_related_scripts
│   │   ├── download_data.py
│   │   └── preprocess_data.py
│   │
│   ├── adaboost
│   │   ├── evaluation
│   │   ├── extract_features.py
│   │   ├── load_data.py
│   │   ├── main.py
│   │   └── train_adaboost.py
│   │
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