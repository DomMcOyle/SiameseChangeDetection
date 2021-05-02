import dataprocessing as dp
import siamese as s
import sklearn.metrics as skm
import keras.models as km
import os
import numpy as np
import config
import configparser

if __name__ == '__main__':
    dataset_name = "BAY AREA"
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    if int(parser["settings"].get("training")) == 1:
        first_img, second_img, labels = dp.load_dataset(dataset_name, parser)
        x_train, y_train = dp.preprocessing(first_img, second_img, labels, parser[dataset_name])

        model = s.siamese_model(x_train[0, 0].shape, int(parser[dataset_name].get("FirstLayerNeurons")), s.euclidean_dist)

        model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                  batch_size=64,
                  epochs=10,
                  verbose=2)
        model.save_weights(config.MODEL_SAVE_PATH + "BAED6410.h5")
        # memo: nomi per i pesi = inizialidataset + loss + batch + epochs
    else:

        # dataset loading and preprocessing
        first_img, second_img, labels = dp.load_dataset(dataset_name, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[dataset_name])

        trained_model = s.siamese_model(x_test[0][0].shape, int(parser[dataset_name].get("FirstLayerNeurons")), s.euclidean_dist)
        trained_model.load_weights("model"+os.sep + "BAED6410.h5")
        distances = trained_model.predict([x_test[:, 0], x_test[:, 1]])

        # converting distances into labels

        prediction = np.where(distances.ravel() < 0.5, config.UNCHANGED_LABEL, config.CHANGED_LABEL)
        cm = skm.confusion_matrix(y_test, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

        metrics = s.get_metrics(cm)
        print("TOTAL OF EXAMPLES: " + str(len(y_test)))
        print("TOTAL OF 1: " + str(sum(cm[1, :])))
        print("TOTAL OF 0: " + str(sum(cm[0, :])))
        print("Metrics:")
        for k in metrics.keys():
            print(k + ": " + str(metrics[k]))

        s.plot_maps(prediction, labels)