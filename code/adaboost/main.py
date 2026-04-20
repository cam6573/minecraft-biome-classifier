import time
import subprocess


def run_step(name, command):
    print("\n" + "=" * 40)
    print(f" STEP: {name}")
    print("=" * 40)

    start = time.time()
    subprocess.run(command, check=True)
    end = time.time()

    print(f"\nCompleted {name} in {end - start:.2f} seconds\n")


def main():
    # feature extraction
    run_step(
        "Feature Extraction",
        ["python3", "code/adaboost/extract_features.py"]
    )

    # training
    run_step(
        "AdaBoost Training",
        ["python3", "code/adaboost/train_adaboost.py"]
    )


if __name__ == "__main__":
    main()