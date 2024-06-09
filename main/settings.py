from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
WEBCAM = 'Webcam'


SOURCES_LIST = [IMAGE, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'plaa.jpeg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'main.png'




# ML Model config
MODEL_DIR =  'C:/Users/MY PC/Desktop/yolov8/yolov8-streamlit-detection-tracking/weights'
DETECTION_MODEL = MODEL_DIR+'yolov8.pkl'
#SEGMENTATION_MODEL = MODEL_DIR+'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
