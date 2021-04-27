import dataprocessing as dp
import siamese as s
import keras.models as km
import os
import numpy as np
import config
import configparser

if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)
    first_img, second_img, labels = dp.load_dataset("BAY AREA", parser)

    """
    x_train, y_train = dp.preprocessing(first_img, second_img, labels)

    model = s.siamese_model(x_train[0, 0].shape, 224, s.euclidean_dist)

    model.fit([x_train[:, 0], x_train[:, 1]], y_train,
              batch_size=64,
              epochs=10,
              verbose=2)
    model.save_weights(config.MODEL_SAVE_PATH + "BAED6410.h5")
    # memo: nomi per i pesi = inizialidataset + loss + batch + epochs

    """
    x_test, y_test = dp.preprocessing(first_img, second_img, labels)
    trained_model = s.siamese_model(x_test[0][0].shape, 224, s.euclidean_dist)
    trained_model.load_weights("model"+os.sep + "BAED6410.h5")
    prediction = trained_model.predict([x_test[:, 0], x_test[:, 1]])
    cm = s.get_confusion_matrix(y_test, prediction)
    metrics = s.get_metrics(cm)
    print("TOTAL OF EXAMPLES: " + str(len(y_test)))
    print("TOTAL OF 1: " + str(sum(cm[1, :])))
    print("TOTAL OF 0: " + str(sum(cm[0, :])))
    print("Metrics:")
    for k in metrics.keys():
        print(k + ": " + str(metrics[k]))
    s.plot_maps(prediction, labels)