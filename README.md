
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
python -m pip install kagglehub pandas numpy matplotlib opencv-python scikit-image tensorflow keras_tuner tensorboard scikit-learn
```

Windows:
```
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install kagglehub pandas numpy matplotlib opencv-python scikit-image tensorflow keras_tuner tensorboard
```

If you run into issues run the following:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```


Download DataSet and Select 600 random images for each biom:
```
python code/data_related_scripts/download_data.py
python code/data_related_scripts/preprocess_data.py 
```

Process HOG Data: 
```
python code/data_related_scripts/HOG_process_data.py
```




#### CNN:

The CNN model should be already trained and under `./model/CNN/minecraft_biome_cnn.keras`


#### Project Structure:
```
.
├── README.md
├── code
│   ├── CNN
│   │   ├── CNN.py
│   │   ├── predict_an_image.py
│   │   └── test_all_image.py
│   └── data_related_scripts
│       ├── CNN
│       │   └── CNN_process_data.py
│       ├── HOG_process_data.py
│       ├── download_data.py
│       └── preprocess_data.py
├── data
│   ├── preprocessed
│   │   ├── test
│   │   ├── training
│   │   └── validation
│   ├── processed
│   │   └── CNN
│   │       ├── test
│   │       ├── training
│   │       └── validation
│   └── raw
│       └── preprocessed_data
├── models
│   └── CNN
│       └── minecraft_biome_cnn.keras
└── resources
```