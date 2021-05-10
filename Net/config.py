DATA_CONFIG_PATH = "net.conf"
MODEL_SAVE_PATH = "model/"
STAT_PATH = "stat/"
# Constants
CHANGED_LABEL = 0
UNCHANGED_LABEL = 1
UNKNOWN_LABEL = 2

# the colors are assigned in ascending order
COLOR_MAP = ['r', 'b', 'y']

# global variables used for model optimization purposes (hyperas)
test_cm = None
val_cm = None
best_score = 0
best_model = None
best_time = 0
train_set = None
test_set = None
train_labels = None
test_labels = None