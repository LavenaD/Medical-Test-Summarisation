import os
from pathlib import Path
TARGET_COL = "impression"
TEXT_COL = "findings"
MODEL_NAME ="google/flan-t5-small"
PROJECT_DIR =os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(PROJECT_DIR).parent
MODEL_DIRECTORY_PATH = os.path.join(BASE_DIR, "models", "artifacts-google")
DATA_DIRECTORY_PATH = os.path.join(BASE_DIR, "data", "processed")
DATA_RAW_DIRECTORY_PATH = os.path.join(BASE_DIR, "data", "raw")
NUM_EPOCHS = 4
# LEARNING_RATE = 8.8e-5
LEARNING_RATE = 5e-4
MAX_TARGET_LENGTH = 1024
MAX_INPUT_LENGTH = 1024