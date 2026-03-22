import kagglehub

path = kagglehub.dataset_download("willowc/minecraft-biomes", output_dir="./data/raw")
print(path)