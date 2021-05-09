import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

import tensorflow as tf
import keras.backend as kb
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks

from hyperas.distributions import uniform, choice

from sklearn.model_selection import train_test_split

import config


def siamese_base_model(input_shape, first_drop, second_drop):
    """
    Function creating the common base for the siamese network
    :param input_shape: the shape of the input for the first layer
    :param first_drop: a float between 0 and 1 indicating first dropout layer's drop rate
    :param second_drop: a float between 0 and 1 indicating second dropout layer's drop rate
    :return: a Model with three dense layers interspersed with two dropout layers
    """
    input_layer = Input(input_shape)
    hidden = Dense(input_shape[0], activation='relu')(input_layer)
    hidden = Dropout(first_drop)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dropout(second_drop)(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    # memo: sperimentare dopo aver ridotto i neuroni di espandere nuovamente
        # a 128 e 512
    return Model(input_layer, hidden)


def siamese_model(train_set, train_label, test_set, test_labels, score_function):
    """
    TODO: MODIFICARE COMMENTI
    Function for creating and testing a siamese model, given the training and the test set as input.
    This function is used inside hyperparam_search, in order to find the model with the best hyperparameters
    :param input_shape: the input shape for the first input layer
    :param score_function: the function to be used for calculating distances. it can be SAM or euclidean_distance
    :return: a compiled siamese model with adam optimization
    """
    # building the net
    input_shape = train_set[0, 0].shape
    first_dropout_rate = {{uniform(0, 0.5)}}
    second_dropout_rate = {{uniform(0, 0.5)}}

    base = siamese_base_model(input_shape, first_dropout_rate, second_dropout_rate)

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    joined_ia = base(input_a)
    joined_ib = base(input_b)

    distance_layer = Lambda(score_function)([joined_ia, joined_ib])
    siamese = Model([input_a, input_b], distance_layer)

    adam = Adam(lr={{uniform(0.0001, 0.01)}})
    siamese.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
    siamese.summary()

    # setting an EarlyStopping callback
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]

    # generating the validation set
    x_train, y_train, x_val, y_val = train_test_split(train_set, train_label, stratify=train_label, test_size=0.2)

    h = siamese.fit([x_train[:, 0], x_train[:, 1]], y_train,
                    batch_size={{choice([32, 64, 128, 256, 512])}},
                    epochs=150,
                    verbose=2,
                    callbacks=callbacks_list,
                    validation_data=([x_val[:, 0], x_val[:, 1]], y_val))

    #TODO: riprendi da riga 160

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
    tp, fn, fp, tn = cm.ravel()
    metrics["overall_accuracy"] = (tn + tp) / (tn + tp + fp + fn)
    metrics["false_positives_num"] = fp
    metrics["false_negatives_num"] = fn
    return metrics


def plot_maps(prediction, label_map):
    """
    function plotting the original label map put beside the predicted label map
    :param prediction: the 1-dim array of predicted classes
    :param label_map: the 2-dim array of shape (height x width) loaded with the dataset
    :return: the plot of the two label map and the whole prediction
    """
    predicted_map = np.reshape(prediction, label_map.shape)

    new_map = np.copy(predicted_map)
    replace_indexes = np.where(label_map == config.UNKNOWN_LABEL)
    new_map[replace_indexes] = config.UNKNOWN_LABEL

    cmap = pltc.ListedColormap(config.COLOR_MAP)
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(predicted_map, cmap=cmap, vmin=0, vmax=2)
    ax1.title.set_text("Total prediction")

    ax2.imshow(new_map, cmap=cmap, vmin=0, vmax=2)
    ax2.title.set_text("Comparable Prediction")

    ax3.imshow(label_map, cmap=cmap, vmin=0, vmax=2)
    ax3.title.set_text("Ground truth")
    plt.show()
    return fig
