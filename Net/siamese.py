import pickle

import numpy as np
import time

import tensorflow as tf
import keras.backend as kb
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks

from hyperas.distributions import uniform, choice
from hyperas import optim
from hyperopt import STATUS_OK, Trials, tpe, STATUS_FAIL

from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from skimage.filters import threshold_otsu
import config


def data():
    """
    Parameterless function to be given to the optim.minimize function of hyperas
    :return: a list containing the training set, the training labels, the test set, the test labels and the selected
             score function
    """
    config.test_cm = []
    config.val_cm = []
    train_set = config.train_set
    train_labels = config.train_labels
    test_set = config.test_set
    test_labels = config.test_labels
    score_function = config.selected_score

    return train_set, train_labels, test_set, test_labels, score_function


def hyperparam_search(train_set, train_labels, test_set, test_labels, score_function, name):
    """
    Function used for training and hyperparameter tuning
    :param train_set: the training set for the model. It is assumed to be an array of preprocessed pixel pairs
    :param train_labels: a 1-dim array containing the training labels. it is assumed it's the same length of train_set
    :param test_set: the test set for the model. The assumption are the same of train_set
    :param test_labels: a 1-dim array containing the test labels for metrics computation. it is assumed it's the same
                        length of test_set
    :param score_function: the function to be used for calculating distances. it can be SAM or euclidean_distance
    :param name: string containing the name of the model to be saved

    :return: the function saves the various statistics on the trials done and the best model retrieved.
        Also it returns the best model and the time of its training

    """
    trials = Trials()
    config.train_set = train_set
    config.train_labels = train_labels
    config.test_set = test_set
    config.test_labels = test_labels
    config.selected_score = score_function
    config.PRED_THRESHOLD = config.AVAILABLE_THRESHOLD[score_function.__name__]

    bs = [32, 64, 128, 256, 512]
    neurons = [256, 512]
    neurons_1 = [128, 256]
    neurons_2 = [64, 128]
    print("Info: BEGINNING SEARCH...")
    best_run, best_model = optim.minimize(model=siamese_model,
                                          data=data,
                                          functions=[siamese_base_model, siamese_model, build_net,
                                                     contrastive_loss, score_function, accuracy],
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=trials
                                          )
    print("Info: SAVING RESULTS...")
    output = open(config.STAT_PATH + name + "_stats.csv", "w")
    output.write("Trials")
    output.write("\ntrial_id, time, epochs, score, loss, val_loss, learning_rate, batch_size, dropout_1," +
                 " dropout_2, layer_1, layer_2, layer_3, " +
                 "test_overall_acc, test_true_positives, test_true_negatives, test_false_positives, " +
                 "test_false_negatives, test_thresh, val_overall_acc, val_true_positives, val_true_negatives," +
                 " val_false_positives, val_false_negatives, val_thresh")

    for trial, test, validation in zip(trials.trials, config.test_cm, config.val_cm):
        if trial['result']['status'] == STATUS_FAIL:
            output.write("\n%s, -, -, -, -, -, %f, %d, %f, %f, %d, %d, %d, FAIL" % (
                trial['tid'],
                trial['misc']['vals']['lr'][0],
                bs[trial['misc']['vals']['batch_size'][0]],
                trial['misc']['vals']['dropout_rate'][0],
                trial['misc']['vals']['dropout_rate_1'][0],
                224, 128, 64
                # neurons[trial['misc']['vals']['layer'][0]],
                # neurons_1[trial['misc']['vals']['layer_1'][0]],
                # neurons_2[trial['misc']['vals']['layer_2'][0]]
            ))
        else:
            tcm = get_metrics(test)
            vcm = get_metrics(validation)
            output.write(
                "\n%s, %d, %d, %f, %f, %f, %f, %d, %f, %f, %d, %d, %d, %f, %d, %d, %d, %d, %f, %f, %d, %d, %d, %d, %f"
                % (trial['tid'],
                   trial['result']['time'],
                   trial['result']['n_epochs'],
                   abs(trial['result']['loss']),
                   trial['result']['cont_loss'],
                   trial['result']['val_cont_loss'],
                   trial['misc']['vals']['lr'][0],
                   bs[trial['misc']['vals']['batch_size'][0]],
                   trial['misc']['vals']['dropout_rate'][0],
                   trial['misc']['vals']['dropout_rate_1'][0],
                   224, 128, 64,
                   # neurons[trial['misc']['vals']['layer'][0]],
                   # neurons_1[trial['misc']['vals']['layer_1'][0]],
                   # neurons_2[trial['misc']['vals']['layer_2'][0]],
                   tcm["overall_accuracy"], tcm["true_positives_num"], tcm["true_negatives_num"],
                   tcm["false_positives_num"], tcm["false_negatives_num"],
                   trial['result']['test_thresh'],
                   vcm["overall_accuracy"], vcm["true_positives_num"], vcm["true_negatives_num"],
                   vcm["false_positives_num"], vcm["false_negatives_num"],
                   trial['result']['val_thresh']
                   ))

    output.write("\nBest model\n")
    best_run['batch_size'] = bs[best_run['batch_size']]
    best_run['layer'] = 224 # neurons[best_run['layer']]
    best_run['layer_1'] = 128 # neurons_1[best_run['layer_1']]
    best_run['layer_2'] = 64 # neurons_2[best_run['layer_2']]
    output.write(str(best_run))
    output.close()

    print("Info: SAVING MODEL (PARAMETERS + WEIGHTS)...")
    best_run.pop('batch_size')
    best_run['score_function'] = score_function
    best_run['margin'] = config.AVAILABLE_MARGIN[score_function.__name__]

    param_file = open(config.MODEL_SAVE_PATH + name + "_param.pickle", "wb")
    pickle.dump(best_run, param_file, pickle.HIGHEST_PROTOCOL)
    param_file.close()

    config.best_model.save_weights(config.MODEL_SAVE_PATH + name + ".h5")

    return config.best_model, config.best_time


def siamese_model(train_set, train_labels, test_set, test_labels, score_function):
    """
    Function for creating and testing a siamese model, given the training and the test set as input.
    This function is used inside hyperparam_search, in order to find the model with the best hyper-parameters
    :param train_set: the training set for the model. It is assumed to be an array of preprocessed pixel pairs
    :param train_labels: a 1-dim array containing the training labels. it is assumed it's the same length of train_set
    :param test_set: the test set for the model. The assumption are the same of train_set
    :param test_labels: a 1-dim array containing the test labels for metrics computation. it is assumed it's the same
                        length of test_set
    :param score_function: the function to be used for calculating distances. it can be SAM or euclidean_distance
    :return: a compiled siamese model with adam optimization
    """
    # building the net
    input_shape = train_set[0, 0].shape
    dropout_rate = {{uniform(0, 0.5)}}
    dropout_rate_1 = {{uniform(0, 0.5)}}
    lr = {{uniform(0.0001, 0.01)}}
    layer = 224 # {{choice([256, 512])}}
    layer_1 = 128 # {{choice([128, 256])}}
    layer_2 = 64 # {{choice([64, 128])}}

    param = {'dropout_rate': dropout_rate,
             'dropout_rate_1': dropout_rate_1,
             'lr': lr,
             'layer': layer,
             'layer_1': layer_1,
             'layer_2': layer_2,
             'score_function': score_function,
             'margin': config.AVAILABLE_MARGIN[score_function.__name__]}

    siamese = build_net(input_shape, param)

    # setting an EarlyStopping callback
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]

    # generating the validation set
    x_train, x_val, y_train, y_val = train_test_split(train_set, train_labels, stratify=train_labels,
                                                      test_size=config.VAL_SPLIT)

    tic = time.time()
    # fitting the model
    try:
        h = siamese.fit([x_train[:, 0], x_train[:, 1]], y_train,
                        batch_size={{choice([32, 64, 128, 256, 512])}},
                        epochs=150,
                        verbose=2,
                        callbacks=callbacks_list,
                        validation_data=([x_val[:, 0], x_val[:, 1]], y_val))
    except:
        print("Error in training")
        config.test_cm.append(np.zeros(shape=(2, 2)))
        config.val_cm.append(np.zeros(shape=(2, 2)))
        return {'status': STATUS_FAIL}

    toc = time.time()
    # printing the best score
    best_epoch_idx = np.nanargmin(h.history['val_loss'])
    # the score returned is the best epoch one
    loss = h.history['loss'][best_epoch_idx]
    score = h.history['val_loss'][best_epoch_idx]
    val_acc = h.history['val_accuracy'][best_epoch_idx]

    print('Score: ' + str(score))
    print('Loss: ' + str(loss))
    print('Validation Accuracy: ' + str(val_acc))
    print('Epochs: ' + str(len(h.history['loss'])))

    # making prediction on the test set
    distances = siamese.predict([test_set[:, 0], test_set[:, 1]])

    test_thresh = threshold_otsu(distances)
    # converting distances into labels
    prediction = np.where(distances.ravel() < test_thresh, config.UNCHANGED_LABEL, config.CHANGED_LABEL)
    cm = skm.confusion_matrix(test_labels, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
    config.test_cm.append(cm)

    # making preditcion on the validation set
    val_distances = siamese.predict([x_val[:, 0], x_val[:, 1]])

    val_thresh = threshold_otsu(val_distances)
    # converting distances into labels
    val_prediction = np.where(val_distances.ravel() < val_thresh, config.UNCHANGED_LABEL, config.CHANGED_LABEL)
    vcm = skm.confusion_matrix(y_val, val_prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
    config.val_cm.append(vcm)

    print('test threshold: ' + str(test_thresh))
    print('val threshold: ' + str(val_thresh))
    print('Best Score: ', str(config.best_score))
    if score < config.best_score:
        config.best_score = score
        config.best_model = siamese
        config.best_time = toc - tic

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']),
            'model': config.best_model, 'time': toc - tic, 'cont_loss': loss, 'val_cont_loss': score,
            'test_thresh': test_thresh, 'val_thresh': val_thresh}


def build_net(input_shape, parameters):
    """
    Function building the architecture for the net. Because of the Lambda layer, the net cant' be serialized as
    a saved_model, so it must be rebuilt each time.

    :param input_shape: the shape for the input layer
    :param parameters: dict, contains the parameter for building the network:
        'dropout_rate': the rate for the first dropout layer
        'dropout_rate_1': the rate for the second dropout layer. The name is given
                            by the hyperas optimization process
        'lr': the learning rate
        'layer': the number of neurons for the first dense layer
        'layer_1': the number of neurons for the second dense layer
        'layer_2': the number of neurons for the third dense layer
        'score_function': the name of the score function selected for the net
        'margin': a float indicating the margin to be used for the contrastive loss function
    :return: a compiled keras model with the given parameters
    """
    base = siamese_base_model(input_shape, parameters['dropout_rate'], parameters['dropout_rate_1'],
                              parameters['layer'], parameters['layer_1'], parameters['layer_2'])

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    joined_ia = base(input_a)
    joined_ib = base(input_b)

    distance_layer = Lambda(parameters['score_function'])([joined_ia, joined_ib])
    siamese = Model([input_a, input_b], distance_layer)

    adam = Adam(lr=parameters['lr'])
    siamese.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
    config.MARGIN = parameters['margin']
    siamese.summary()
    return siamese


def siamese_base_model(input_shape, first_drop, second_drop, first_layer, second_layer, third_layer):
    """
    Function creating the common base for the siamese network
    :param input_shape: the shape of the input for the first layer
    :param first_drop: a float between 0 and 1 indicating first dropout layer's drop rate
    :param second_drop: a float between 0 and 1 indicating second dropout layer's drop rate
    :param first_layer: a positive integer indicating the number of neurons of the first layer
    :param second_layer: a positive integer indicating the number of neurons of the second layer
    :param third_layer: a positive integer indicating the number of neurons of the third layer
    :return: a Model with three dense layers interspersed with two dropout layers
    """
    input_layer = Input(input_shape)
    hidden = Dense(first_layer, activation='relu')(input_layer)
    hidden = Dropout(first_drop)(hidden)
    hidden = Dense(second_layer, activation='relu')(hidden)
    hidden = Dropout(second_drop)(hidden)
    hidden = Dense(third_layer, activation='relu')(hidden)
    hidden = Dense(512, activation='sigmoid')(hidden)
    return Model(input_layer, hidden)


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
    # dot must be bounded since some values could a little bit more than 1 or less than -1
    dot = kb.maximum(dot, -1)
    dot = kb.minimum(dot, 1)
    return tf.math.acos(dot)


def contrastive_loss(y_true, y_pred):
    """
    Function implementing the contrastive loss for the training phase
    :param y_true: the actual value for the pixel's class (1 not changed = same class, 0 changed = different class)
    :param y_pred: the predicted value for the pixel's class
    :return: the value of the contrastive loss. The margin, positive value which helps to make largely
    dissimilar pairs to count toward the loss computation, is stored in config.py
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = kb.square(y_pred)
    square_margin = kb.square(kb.maximum(config.MARGIN - y_pred, 0))
    return kb.mean(y_true * square_pred + (1 - y_true) * square_margin)


def accuracy(y_true, y_pred):
    """
    Function computing the training accuracy
    :param y_true: the actual value for the pixel's class (1 not changed = same class, 0 changed = different class)
    :param y_pred: the predicted value for the pixel's class
    :return: the accuracy considering y_pred = 1 <=> y_pred<0.5
    """
    return kb.mean(kb.equal(y_true, kb.cast(y_pred < config.PRED_THRESHOLD, y_true.dtype)))


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
    metrics["true_negatives_num"] = tn
    metrics["true_positives_num"] = tp
    return metrics
