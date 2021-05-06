import dataprocessing as dp
import siamese as s
import sklearn.metrics as skm
import keras.models as km
import os
import numpy as np
import config
import configparser

if __name__ == '__main__':
    dataset_name = "SANTA BARBARA"
    model_name = "BASAM6410.h5"
    distance = s.SAM

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    if int(parser["settings"].get("training")) == 1:
        first_img, second_img, labels = dp.load_dataset(dataset_name, parser)
        x_train, y_train = dp.preprocessing(first_img, second_img, labels, parser[dataset_name])

        model = s.siamese_model(x_train[0, 0].shape, int(parser[dataset_name].get("FirstLayerNeurons")), distance)

        model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                  batch_size=64,
                  epochs=10,
                  verbose=2)
        model.save_weights(config.MODEL_SAVE_PATH + model_name)
        # memo: nomi per i pesi = inizialidataset + loss + batch + epochs
    else:

        # dataset loading and preprocessing
        first_img, second_img, labels = dp.load_dataset(dataset_name, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[dataset_name])

        # model loading and prediction
        trained_model = s.siamese_model(x_test[0][0].shape, int(parser[dataset_name].get("FirstLayerNeurons")), distance)
        trained_model.load_weights(config.MODEL_SAVE_PATH + model_name)
        distances = trained_model.predict([x_test[:, 0], x_test[:, 1]])

        # converting distances into labels

        prediction = np.where(distances.ravel() < 0.5, config.UNCHANGED_LABEL, config.CHANGED_LABEL)
        cm = skm.confusion_matrix(y_test, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

        # printing the metrics
        metrics = s.get_metrics(cm)
        file = open(config.STAT_PATH + dataset_name+"_on_"+model_name+".txt", "w")
        file.write("TOTAL OF EXAMPLES: " + str(len(y_test))+"\n")
        file.write("TOTAL OF 1: " + str(sum(cm[1, :]))+"\n")
        file.write("TOTAL OF 0: " + str(sum(cm[0, :]))+"\n")
        file.write("Metrics:"+"\n")
        for k in metrics.keys():
            file.write(k + ": " + str(metrics[k])+"\n")
        file.close()

        fig = s.plot_maps(prediction, dp.refactor_labels(np.asarray(labels)[0], parser[dataset_name]))
        fig.savefig(config.STAT_PATH + dataset_name+"_on_"+model_name+".png")
