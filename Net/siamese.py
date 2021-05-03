import numpy as np
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model
import keras.backend as kb
import tensorflow as tf
import sklearn.metrics as skm
import config
import matplotlib.pyplot as plt
import matplotlib.colors as pltc


def siamese_base_model(input_shape, first_layer_dim):
    """
    Function creating the common base for the siamese network
    :param input_shape: the shape of the input for the first layer
    :param first_layer_dim: a positive integer indicating the number of neurons for the fist layer
    :return: a Model with three dense layers
    """
    input_layer = Input(input_shape)
    hidden = Dense(first_layer_dim, activation='relu')(input_layer)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    # memo: sperimentare dopo aver ridotto i neuroni di espandere nuovamente
        # a 128 e 512
    return Model(input_layer, hidden)


def siamese_model(input_shape, first_layer_dim, score_function):
    """
    Function returning the compiled siamese model
    this is a temporary func, since it doesn't allow automatic hyperparameters tuning
    :param input_shape: the input shape for the first input layer
    :param first_layer_dim: a positive integer indicating the number of neurons for the fist layer
    :param score_function: the loss function to be used for training between SAM and euclidean_distance
    :return: a compiled siamese model with adam optimization
    """
    base = siamese_base_model(input_shape, first_layer_dim)
    input_a = Input(input_shape)
    input_b = Input(input_shape)

    joined_ia = base(input_a)
    joined_ib = base(input_b)

    distance_layer = Lambda(score_function)([joined_ia, joined_ib])
    siamese = Model([input_a, input_b], distance_layer)
    siamese.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
    siamese.summary()
    return siamese


def euclidean_dist(tens):
    """
    Function defining the euclidean distance to be applied on
    the result of the Siamese Branches
    :param tens: the two tensors to be compared
    :return: the euclidean distance between the tensors sqrt((t1^2-t2^2))
    """
    (x, y) = tens
    squared_sum = kb.sum(kb.square(x - y), axis=1, keepdims=True)
    return kb.sqrt(kb.maximum(squared_sum, kb.epsilon()))


def SAM(tens):
    """
    Function defining the Spectral Angle Mapper score to be applied
    on the result of the Siamese Branches
    :param tens: the two tensors (spectral signature of each pixel) to be compared
    :return: the SAM score between the tensors as arccos(t1xt2/norm2(t1)*norm2(t2))
    """
    (x, y) = tens
    xnorm = kb.l2_normalize(x, axis=1)
    ynorm = kb.l2_normalize(y, axis=1)
    dot = kb.sum(xnorm * ynorm, axis=1, keepdims=True)
    return tf.math.acos(dot)


def contrastive_loss(y_true, y_pred, margin=1):
    """
    Function implementing the contrastive loss for the training phase
    :param y_true: the actual value for the pixel's class (1 not changed = same class, 0 changed = different class)
    :param y_pred: the predicted value for the pixel's class
    :param margin: positive value which helps to make largely dissimilar pairs to count toward the loss computation
    :return: the value of the contrastive loss
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = kb.square(y_pred)
    square_margin = kb.square(kb.maximum(margin - y_pred, 0))
    return kb.mean(y_true * square_pred + (1 - y_true) * square_margin)


def accuracy(y_true, y_pred):
    """
    Function computing the training accuracy
    :param y_true: the actual value for the pixel's class (1 not changed = same class, 0 changed = different class)
    :param y_pred: the predicted value for the pixel's class
    :return: the accuracy considering y_pred = 1 <=> y_pred<0.5
    """
    return kb.mean(kb.equal(y_true, kb.cast(y_pred < 0.5, y_true.dtype)))


def get_metrics(cm):
    """
    function computing the metrics for the model, given the sklearn confusion matrix in input
    :param cm: the sklearn confusion matrix created from the prediction
    :return: a dictionary containing the currently implemented metrics
    """
    metrics = dict()
    tn, fp, fn, tp = cm.ravel()
    metrics["overall_accuracy"] = (tn + tp) / (tn + tp + fp + fn)
    metrics["false_positives_num"] = fp
    metrics["false_negatives_num"] = fn
    return metrics


def plot_maps(prediction, label_map):
    """
    function plotting the original label map put beside the predicted label map
    :param prediction: the 2-dim array of labels
    :param label_map: the 2-dim array of shape (height x width) loaded with the dataset
    :param thresh: float indicating the threshold below which a prediction should be labeled as "1" (not-changed)
    :return: the plot of the two label map
    """
    new_map = np.copy(label_map)
    replace_indexes = np.where(new_map != config.UNKNOWN_LABEL)
    new_map[replace_indexes] = prediction
    cmap = pltc.ListedColormap(config.COLOR_MAP)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(label_map, cmap=cmap)
    ax1.title.set_text("Ground truth")
    ax2.imshow(new_map, cmap=cmap)
    ax2.title.set_text("Prediction")

    plt.show()
    return fig
