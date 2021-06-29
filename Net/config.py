from numpy import pi

DATA_CONFIG_PATH = "net.conf"
MODEL_SAVE_PATH = "model/"
STAT_PATH = "stat/"
# Constants
CHANGED_LABEL = 0
UNCHANGED_LABEL = 1
UNKNOWN_LABEL = 2
AVAILABLE_THRESHOLD = {'euclidean_dist': 0.5, 'SAM': pi/4}
AVAILABLE_MARGIN = {'euclidean_dist': 1, 'SAM': pi/2}
PRED_THRESHOLD = 0.5
MARGIN = 1
VAL_SPLIT = 0.2
MAX_EVALS = 30

# the colors are assigned in ascending order
# r = 0 => changed, b = 1 => unchanged, y = 2 => unknown
COLOR_MAP = ['r', 'b', 'y']

# global variables used for model optimization purposes (hyperas)
test_cm = None
val_cm = None
best_score = float("inf")
best_model = None
best_time = 0
train_set = None
test_set = None
train_labels = None
test_labels = None
selected_score = None
batch_size = None
max_dropout = 0
neurons = None
neurons_1 = None
neurons_2 = None
fourth_layer = False
