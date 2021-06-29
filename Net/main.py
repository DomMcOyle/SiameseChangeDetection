import time

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
    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    train_set = parser["settings"].get("train_set")
    test_set = parser["settings"].get("test_set")
    model_name = parser["settings"].get("model_name")

    print("Selected train dataset: " + train_set)
    print("Selected test dataset: " + test_set)
    print("Selected model: " + model_name)

    if parser["settings"].get("distance") == "ED":
        hyperas_sett = "hyperas settings ED"
        distance_func = s.euclidean_dist
    elif parser["settings"].get("distance") == "SAM":
        hyperas_sett = "hyperas settings SAM"
        distance_func = s.SAM
    else:
        raise NotImplementedError("Error: DISTANCE FUNCTION " + parser["settings"].get("distance") + " NOT IMPLEMENTED")

    if parser["settings"].getboolean("training") is True:
        # loading the pairs
        train_a_img, train_b_img, train_labels, train_names = dp.load_dataset(train_set, parser)
        test_a_img, test_b_img, test_labels, test_names = dp.load_dataset(test_set, parser)

        # executing preprocessing
        x_train, y_train = dp.preprocessing(train_a_img, train_b_img, train_labels, parser[train_set],
                                            keep_unlabeled=False,
                                            apply_rescaling=parser["settings"].getboolean("apply_rescaling"))
        x_test, y_test = dp.preprocessing(test_a_img, test_b_img, test_labels, parser[test_set],
                                          keep_unlabeled=True,
                                          apply_rescaling=parser["settings"].getboolean("apply_rescaling"))

        print("Info: STARTING HYPERSEARCH PROCEDURE")
        model, run = s.hyperparam_search(x_train, y_train, x_test, y_test, distance_func, model_name,
                                         parser[hyperas_sett])

    else:
        # declaring placeholder to be printed when no fine tuning is executed
        pseudo_qty = "no fine tuning"
        extraction_time = "-"
        fine_time = "-"
        loss = "-"
        val_loss = "-"
        val_acc = "-"
        epochs = "-"
        pseudo_accuracy = "-"
        pseudo_accuracy_corr = "-"

        percentage = float(parser["settings"].get("pseudo_percentage"))
        radius = int(parser["settings"].get("pseudo_radius"))

        # dataset and model loading
        first_img, second_img, labels, names = dp.load_dataset(test_set, parser)
        x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[test_set],
                                          keep_unlabeled=True,
                                          apply_rescaling=parser["settings"].getboolean("apply_rescaling"))

        # parameters loading
        print("Info: LOADING THE MODEL...")
        param_file = open(config.MODEL_SAVE_PATH + model_name + "_param.pickle", "rb")
        parameters = pickle.load(param_file)
        param_file.close()
        model = s.build_net(x_test[0, 0].shape, parameters)

        i = 0
        # The model will be tuned fresh for each image
        # and each image must be extracted from the preprocessed matrix
        for lab in labels:
            model.load_weights(config.MODEL_SAVE_PATH + model_name + ".h5")
            pairs = x_test[i:i+lab.size, :]
            img_label = y_test[i:i+lab.size]

            # Fine tuning phase
            # loading the dictionary containing distances and the threshold
            if int(parser["settings"].get("fine_tuning")) >= 0:
                # loading the dictionary containing distances, threshold and shape
                pseudo_file = open(parser[test_set].get("pseudoPath") + "/" + names[i] + ".pickle", "rb")
                pseudo_dict = pickle.load(pseudo_file)
                pseudo_file.close()

                # generating spatial corrected pseudo_labels for metrics computing
                pseudo_truth = np.where(pseudo_dict["distances"] > pseudo_dict["threshold"], config.CHANGED_LABEL,
                                        config.UNCHANGED_LABEL)
                pseudo_truth = pu.spatial_correction(np.reshape(pseudo_truth, pseudo_dict["shape"])).ravel()

                if int(parser["settings"].get("fine_tuning")) == 0:

                    pseudo_qty = "all"
                    toc_extraction = 0
                    tic_extraction = 0
                    best_data = np.arange(0, len(pseudo_dict["distances"]))
                    pseudo = pseudo_truth

                elif int(parser["settings"].get("fine_tuning")) == 1:

                    pseudo_qty = str(percentage*100)+"%"
                    tic_extraction = time.time()
                    best_data, pseudo = pu.labels_by_percentage(pseudo_dict, percentage)
                    toc_extraction = time.time()
                elif int(parser["settings"].get("fine_tuning")) == 2:

                    pseudo_qty = "r=" + str(radius)
                    tic_extraction = time.time()
                    best_data, pseudo = pu.labels_by_neighborhood(pseudo_dict, radius)
                    toc_extraction = time.time()
                else:
                    raise ValueError("Error: FINE TUNING CHOICE " + parser["settings"].get("fine_tuning")
                                     + " NOT IMPLEMENTED")

                extraction_time = toc_extraction - tic_extraction
                print("Info: PERFORMING FINE TUNING...")
                model, loss, val_loss, val_acc, epochs, fine_time = s.fine_tuning(model,
                                                                                  parameters['batch_size'],
                                                                                  pairs[best_data], pseudo)

            # prediction
            print("Info: EXECUTING PREDICTION OF " + names[i] + " " + str(i+1) + "/" + str(len(labels)))
            tic_prediction = time.time()
            distances = model.predict([pairs[:, 0], pairs[:, 1]])
            toc_prediction = time.time()
            prediction_time = toc_prediction - tic_prediction

            # computing threshold and turning distances into labels
            threshold = threshold_otsu(distances)
            prediction = np.where(distances.ravel() > threshold,
                                  config.CHANGED_LABEL, config.UNCHANGED_LABEL)

            print("Info: COMPUTING THE METRICS...")
            # print the heatmap
            im = plt.imshow(distances.reshape(lab.shape), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_" + pseudo_qty
                        + "_heatmap.png", dpi=300, bbox_inches='tight')

            # 1. confusion matrix
            cm = skm.confusion_matrix(img_label, prediction, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            # 2. getting the metrics
            metrics = s.get_metrics(cm)

            print("Info: SAVING THE " + str(i+1) + "° RESULT")
            file = open(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_" + pseudo_qty
                        + ".csv", "w")

            # 3. printing column names
            file.write("total_examples, threshold")
            for k in metrics.keys():
                file.write(", " + k)
            file.write(", prediction_time")
            for k in metrics.keys():
                file.write(", " + k + "_correction")
            file.write(", correction_time, pseudo_qty, extraction_time," +
                       " ft_epochs, ft_time, ft_loss, ft_val_loss, ft_val_acc, pseudo_acc, pseudo_acc_corrected")
            file.write("\n %d, %f" % (len(img_label), threshold))

            # 4. printing metrics
            for k in metrics.keys():
                file.write(", " + str(metrics[k]))
            file.write(", " + str(prediction_time))

            # 5. saving the map plot
            # a. the prediction is reshaped as a 2-dim array
            lmap = np.reshape(prediction, lab.shape)
            # b. label is refactored singularly in order to provide coherent ground truth
            ground_t = dp.refactor_labels(lab, parser[test_set])
            # c. the maps are plotted with the appropriate function
            fig = pu.plot_maps(lmap, ground_t)
            fig.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_" + pseudo_qty
                        + ".png", dpi=300, bbox_inches='tight')

            print("Info: EXECUTING SPATIAL CORRECTION...")
            # replying steps 1, 2, 4 and 5 after the spacial correction
            tic_correction = time.time()
            corrected_map = pu.spatial_correction(lmap)
            toc_correction = time.time()
            correction_time = toc_correction - tic_correction

            print("Info: GETTING AND SAVING THE METRICS AFTER SC...")
            sccm = skm.confusion_matrix(img_label, corrected_map.ravel(),
                                        labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

            scmetrics = s.get_metrics(sccm)

            # if fine tuning is enabled, accuracy with respect to the pseudo labels is computed
            if int(parser["settings"].get("fine_tuning")) >= 0:
                pseudocm = skm.confusion_matrix(pseudo_truth, prediction,
                                                labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
                pseudocm_corrected = skm.confusion_matrix(pseudo_truth, corrected_map.ravel(),
                                                          labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
                pseudo_accuracy = s.get_metrics(pseudocm)["overall_accuracy"]
                pseudo_accuracy_corr = s.get_metrics(pseudocm_corrected)["overall_accuracy"]

            for k in scmetrics.keys():
                file.write(", " + str(scmetrics[k]))
            file.write(", " + str(correction_time) + ", " + pseudo_qty + ", " + str(extraction_time) +
                       ", " + str(epochs) + ", " + str(fine_time) + ", " + str(loss) + ", " + str(val_loss) +
                       ", " + str(val_acc) + ", " + str(pseudo_accuracy) + ", " + str(pseudo_accuracy_corr))
            file.write("\n")
            file.close()

            scfig = pu.plot_maps(corrected_map, ground_t)
            scfig.savefig(config.STAT_PATH + test_set + "_" + names[i] + "_on_" + model_name + "_" + pseudo_qty
                          + "_corrected.png", dpi=300, bbox_inches='tight')

            i = i + lab.size
