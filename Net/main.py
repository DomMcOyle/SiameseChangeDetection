import dataprocessing as dp
import siamese as s
import predutils as pu
import sklearn.metrics as skm
import numpy as np
import config
import configparser
import pickle
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

if __name__ == '__main__':
    train_set = "BAY AREA"
    test_set = "SANTA BARBARA"
    model_name = "BASAMotsu"
    distance = s.SAM

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    if int(parser["settings"].get("training")) == 1:
        # loading the pairs
        train_a_img, train_b_img, train_labels = dp.load_dataset(train_set, parser)
        test_a_img, test_b_img, test_labels = dp.load_dataset(test_set, parser)

        # executing preprocessing
        x_train, y_train = dp.preprocessing(train_a_img, train_b_img, train_labels, parser[train_set], False)
        x_test, y_test = dp.preprocessing(test_a_img, test_b_img, test_labels, parser[test_set], True)

        print("Info: STARTING HYPERSEARCH PROCEDURE")
        model, run = s.hyperparam_search(x_train, y_train, x_test, y_test, distance, model_name)

    else:

        # dataset and model loading
        first_img, second_img, labels = dp.load_dataset(test_set, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[test_set], True)

        # parameters loading
        print("Info: LOADING THE MODEL...")
        param_file = open(config.MODEL_SAVE_PATH + model_name + "_param.pickle", "rb")
        parameters = pickle.load(param_file)
        model = s.build_net(x_test[0, 0].shape, parameters)

        model.load_weights(config.MODEL_SAVE_PATH + model_name + ".h5")

        # prediction
        print("Info: EXECUTING PREDICTIONS...")
        distances = model.predict([x_test[:, 0], x_test[:, 1]])

        # converting distances into labels
        config.PRED_THRESHOLD = threshold_otsu(distances)
        print(config.PRED_THRESHOLD)
        prediction = np.where(distances.ravel() < config.PRED_THRESHOLD, config.UNCHANGED_LABEL, config.CHANGED_LABEL)

        print("Info: SAVING THE RESULTS...")
        i = 0
        for lab in labels:
            # get the image
            img = prediction[i:i+lab.size]
            dist = distances[i:i+lab.size]

            # print the heatmap
            im = plt.imshow(dist.reshape(lab.shape), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(config.STAT_PATH + test_set+"_on_"+model_name+"_heatmap.png", dpi=300, bbox_inches='tight')

            # confusion matrix
            cm = skm.confusion_matrix(y_test, img, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            # getting the metrics
            metrics = s.get_metrics(cm)

            file = open(config.STAT_PATH + test_set+"_on_"+model_name+".csv", "w")
            # printing columns names
            file.write("total_examples")
            for k in metrics.keys():
                file.write(", " + k)
            file.write(", threshold")
            file.write("\n" + str(len(y_test)))

            # printing metrics
            for k in metrics.keys():
                file.write(", " + str(metrics[k]))
            file.write(", " + str(config.PRED_THRESHOLD))
            file.write("\n" + str(len(y_test)))

            # saving the map plot
            lmap = np.reshape(img, lab.shape)
            ground_t = dp.refactor_labels(lab, parser[test_set])
            fig = pu.plot_maps(lmap, ground_t)
            fig.savefig(config.STAT_PATH + test_set+"_on_"+model_name+".png", dpi=300, bbox_inches='tight')

            # spacial correction + metrics + map
            corrected_map = pu.spatial_correction(lmap)
            sccm = skm.confusion_matrix(y_test, corrected_map.ravel(), labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
            scmetrics = s.get_metrics(sccm)

            for k in scmetrics.keys():
                file.write(", " + str(scmetrics[k]))
            file.write(", " + str(config.PRED_THRESHOLD))
            file.write("\n")
            file.close()
            scfig = pu.plot_maps(corrected_map, ground_t)
            scfig.savefig(config.STAT_PATH + test_set+"_on_"+model_name+"_corrected.png", dpi=300, bbox_inches='tight')

            i = i + lab.size
