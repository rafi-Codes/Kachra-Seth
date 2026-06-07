"""Constants used throughout the project."""

# Model constants
IMAGE_SIZE = 384
MAX_COUNT = 500
MIN_COUNT = 5
MC_DROPOUT_PASSES = 15

# Confidence thresholds
CONFIDENCE_HIGH = 0.90
CONFIDENCE_MODERATE = 0.70
CONFIDENCE_LOW = 0.50

# Error messages
ERROR_NO_STACK = "no_stack_found"
ERROR_HIGH_DENSITY = "high_density_stack"
ERROR_LOW_QUALITY = "low_image_quality"

# Uncertainty levels
UNCERTAINTY_LOW = "low"
UNCERTAINTY_MEDIUM = "medium"
UNCERTAINTY_HIGH = "high"

# Blur detection
BLUR_THRESHOLD_LAPLACIAN = 100

# Data splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Default categories
DEFAULT_CATEGORIES = [
    "banknotes",
    "books", 
    "papers",
    "tiles",
    "cards",
    "plates"
]

# Performance targets
TARGET_MAE_SMALL = 3  # for 5-50 items
TARGET_MAE_MEDIUM = 5  # for 51-150 items
TARGET_MAPE_LARGE = 0.08  # 8% for 151-500 items
TARGET_CONFIDENCE_ECE = 0.05
TARGET_WITHIN_5PCT_ACCURACY = 0.85
TARGET_INFERENCE_LATENCY_MS = 200
