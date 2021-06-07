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
    model_name = "BAEDotsu"
    distance_func = s.euclidean_dist

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    if int(parser["settings"].get("training")) == 1:
        # loading the pairs
        train_a_img, train_b_img, train_labels, train_names = dp.load_dataset(train_set, parser)
        test_a_img, test_b_img, test_labels, test_names = dp.load_dataset(test_set, parser)

        # executing preprocessing
        x_train, y_train = dp.preprocessing(train_a_img, train_b_img, train_labels, parser[train_set], False)
        x_test, y_test = dp.preprocessing(test_a_img, test_b_img, test_labels, parser[test_set], True)

        if distance_func is s.SAM:
            hyperas_sett = "hyperas settings SAM"
        elif distance_func is s.euclidean_dist:
            hyperas_sett = "hyperas settings ED"
        else:
            raise NotImplementedError("Error: DISTANCE FUNCTION NOT IMPLEMENTED")

        print("Info: STARTING HYPERSEARCH PROCEDURE")
        model, run = s.hyperparam_search(x_train, y_train, x_test, y_test, distance_func, model_name, parser[hyperas_sett])

    else:
        # dataset and model loading
        first_img, second_img, labels, names = dp.load_dataset(test_set, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[test_set], keep_unlabeled=True)

        # parameters loading
        print("Info: LOADING THE MODEL...")
        param_file = open(config.MODEL_SAVE_PATH + model_name + "_param.pickle", "rb")
        parameters = pickle.load(param_file)

        model = s.build_net(x_test[0, 0].shape, parameters)

        i = 0
        # The model will be tuned fresh for each image
        # and each image must be extracted from the preprocessed matrix
        for lab in labels:
            model.load_weights(config.MODEL_SAVE_PATH + model_name + ".h5")
            img_a = x_test[i:i+lab.size, 0]
            img_b = x_test[i:i+lab.size, 1]
            img_label = y_test[i:i+lab.size]

            # Fine tuning phase
            # The SAM function returned the best pseudo labels, so it will always be used
            pseudo, thresh = pu.pseudo_labels(img_a, img_b, s.SAM)
            config.PRED_THRESHOLD = config.AVAILABLE_THRESHOLD[distance_func.__name__]
            model = s.fine_tuning(model, parameters['batch_size'], x_test[i:i+lab.size, :], pseudo)

            # prediction
            print("Info: EXECUTING PREDICTION OF " + names[i] + " " + str(i+1) + "/" + str(len(labels)))
            distances = model.predict([img_a, img_b])

            # computing threshold and turning distances into labels
            threshold = threshold_otsu(distances)
            prediction = np.where(distances.ravel() > threshold,
                                  config.CHANGED_LABEL, config.UNCHANGED_LABEL)
            print("Info: SAVING THE " + str(i+1) + "° RESULT")

            # print the heatmap
            im = plt.imshow(distances.reshape(lab.shape), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_heatmap.png",
                        dpi=300, bbox_inches='tight')

            # 1. confusion matrix
            cm = skm.confusion_matrix(img_label, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            # 2. getting the metrics
            metrics = s.get_metrics(cm)

            file = open(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + ".csv", "w")

            # 3. printing column names
            file.write("total_examples")
            for k in metrics.keys():
                file.write(", " + k)
            file.write(", threshold")
            file.write("\n" + str(len(img_label)))

            # 4. printing metrics
            for k in metrics.keys():
                file.write(", " + str(metrics[k]))
            file.write(", " + str(threshold))
            file.write("\n" + str(len(img_label)))

            # 5. saving the map plot
            # a. the prediction is reshaped as a 2-dim array
            lmap = np.reshape(prediction, lab.shape)
            # b. label is refactored singularly in order to provide coherent ground truth
            ground_t = dp.refactor_labels(lab, parser[test_set])
            # c. the maps are plotted with the appropriate function
            fig = pu.plot_maps(lmap, ground_t)
            fig.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + ".png",
                        dpi=300, bbox_inches='tight')

            # replying steps 1, 2, 4 and 5 after the spacial correction
            corrected_map = pu.spatial_correction(lmap)
            sccm = skm.confusion_matrix(img_label, corrected_map.ravel(),
                                        labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
            scmetrics = s.get_metrics(sccm)
            for k in scmetrics.keys():
                file.write(", " + str(scmetrics[k]))
            file.write(", " + str(threshold))
            file.write("\n")
            file.close()
            scfig = pu.plot_maps(corrected_map, ground_t)
            scfig.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_corrected.png",
                          dpi=300, bbox_inches='tight')

            i = i + lab.size
