import numpy as np
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model
import keras.backend as kb
import tensorflow as tf


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


def euclidean_dist(tens):
    """
    Function defining the euclidean distance to be applied on
    the result of the Siamese Branches
    :param tens: the two tensors to be compared
    :return: the euclidean distance between the tensors (t1^2-t2^2)
    """
    (x, y) = tens
    squared_sum = kb.sum(kb.square(x - y), axis=1, keepdims=True)
    return kb.sqrt(kb.maximum(squared_sum, kb.epsilon()))

 #TODO: inserire la SAM
def SAM(tens):
   return


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
    return kb.mean(kb.equal(y_true, kb.cast(y_pred < 0.5, y_true.dtype)))


def siamese_model(input_shape, first_layer_dim):
    """
    Function returning the compiled siamese model
    this is a temporary func, since it doesn't allow automatic hyperparameters tuning
    :param input_shape: the input shape for the first input layer
    :param first_layer_dim: a positive integer indicating the number of neurons for the fist layer
    :return: a compiled siamese model with adam optimization
    """
    base = siamese_base_model(input_shape, first_layer_dim)
    input_a = Input(input_shape)
    input_b = Input(input_shape)

    joined_ia = base(input_a)
    joined_ib = base(input_b)

    distance_layer = Lambda(euclidean_dist)([joined_ia, joined_ib])
    siamese = Model([input_a, input_b], distance_layer)
    siamese.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
    siamese.summary()
    return siamese
