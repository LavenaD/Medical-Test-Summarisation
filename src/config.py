import os
TARGET_COL = "impression"
TEXT_COL = "findings"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRECTORY_PATH = os.path.join(BASE_DIR, "models","compressed-artifacts-google")
DATA_DIRECTORY_PATH = os.path.join(BASE_DIR, "data", "processed")
NUM_EPOCHS = 4
LEARNING_RATE = 8.8e-5
