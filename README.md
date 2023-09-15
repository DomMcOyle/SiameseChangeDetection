# Change Detection with Siamese Network and Transfer Learning

Repository for the paper "Siamese networks with transfer learning for change detection in sentinel-2 images" 
(G. Andresini, A. Appice, D. Dell’Olio, D. Malerba, In Proceedings of the International Conference of the Italian Association for Artificial Intelligence, Virtual Event, 1–3 December 2021; Springer: Berlin/Heidelberg, Germany, 2022; pp. 478–489)

Please cite our work if you find it useful for your research and work.

```
@inproceedings{10.1007/978-3-031-08421-8_33,
author = {Andresini, Giuseppina and Appice, Annalisa and Dell’Olio, Domenico and Malerba, Donato},
title = {Siamese Networks With Transfer Learning For Change Detection In Sentinel-2 Images},
year = {2021},
isbn = {978-3-031-08420-1},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
doi = {10.1007/978-3-031-08421-8_33},
booktitle = {AIxIA 2021 – Advances in Artificial Intelligence: 20th International Conference of the Italian Association for Artificial Intelligence, Virtual Event, December 1–3, 2021, Revised Selected Papers},
pages = {478–489},
numpages = {12}
}

```

The dataset used is available at: shorturl.at/gtKO1
## Repository structure

    
        .
        ├── Net       	            # contains the files for the computation (files .txt, .csv, .py)
        │    ├ data         		# dataset location
        │    │  ├ bayarea/pseudo    # contains the selected pseudo-labels for the pair "Bay Area"
        │    │  ├ oneratest/pseudo  # contains the selected pseudo-labels for the Onera dataset pairs (test)
        │    │  ├ oneratrain/pseudo # contains the selected pseudo-labels for the Onera dataset pairs (train)
        │    │  └ barbara/pseudo    # contains the selected pseudo-labels for the pair "Santa Barbara"
        │    ├ model                # model saving and loading location
        │    │   └ model.old 		# contains the models produced for the paper
        │    ├ stat           		# metrics and statistics saving location
        │    │  └ stat.old          # contains models .csv and .png maps
        │    ├ config.py            # contains 'internal' constants for the processing
        │    ├ contents.txt         # text file detailing the contents of the folders stat, model and data     
        │    ├ dataprocessing.py    # contains dataset loading and pre-processing functions
        │    ├ labelbyneigh_generation.py # proximity pseudo-labels maps generation script
        │    ├ labelbyperc_generation.py  # percentage pseudo-labels maps generation script
        │    ├ main.py              # contains main training and testing script
        │    ├ net.conf             # config file for the different combinations of datasets and algorithms
        │    ├ predutils.py         # contains prediction support functions
        │    ├ requirements.txt     # contains dependencies requirements
        │    └ siamese.py           # contains training and fine-tuning function definitions
        ├── res         		    # contains a description of the onera dataset (unused), some experiment's stats and various resources
        └── README.md

Both models and statistics files in model.old and stat.old are split in subfolder system. Each folder refers to an experiment.

### Models
Each trained model is saved in two types of files:

    modelname.h5              # contains the weight learned during training
    modelname_param.pickle    # contains a serialized dictionary with the useful info to build the net architecture

Generally, the name of the models was obtained by associating the initials of the dataset ("BA," "SB," "OTR," "OTE") with the distance abbreviation ("ED" or "SAM")
plus additional substrings to indicate certain characteristics.
The dictionary associated with each model has the following fields:

    'dropout_rate': float - dropout rate for the first dropout layer
    'dropout_rate_1': float - dropout rate for the second dropout layer
    'lr': float - learning rate
    'layer': int - neuron number for the first dense layer
    'layer_1': int - neuron number for the second dense layer
    'layer_2': int - neuron number for the third dense layer
    'score_function': function - distance function selected for the net
    'margin': float - margin for the contrastive loss
    'fourth_layer': boolean - indicates whether to add a fourth 512-neurons layer with sigmoid activation 
	'batch_size': int - training batch size

NB: All experiments prior to using the Otsu threshold have an outdated parameter dictionary. 
    To use them, it is necessary to rerun the training or change them manually.
### Pseudo-labels
Each derived pseudo-label file is saved as a .pickle file containing a serialized dictionary, containing the following fields:

    'threshold': float - threshold value to be used to convert distances to pseudo-labels
    'distances': 1d array of floats - vinearized map of calculated distances 
    'shape': 2d tuple of ints - original size of the image pair
With the returned values, it is then possible to reconstruct the pseudo-label map at a later time after reloading the file. The latter must have the same name as the image pair it refers to.

### Evaluation

The following files were produced for each trial based on the data collected:

    modelname_stats.csv
    # report containing metrics and learning information with hyperparameter optimization (hyperas trials)
    
    dataset[_image]_on_modelname_[0.x/r=y/no fine tuning/all].csv
    # report containing metrics on testing "dataset" on model "modelname" possibly without (no fine tuning) 
    # or with fine tuning with all pseudo-labels (all), a percentage (x%) or extracted by neighborhood (radius y)

    dataset[_image]_on_modelname_[0.x/r=y/no fine tuning/all].png
    # image containing the inferred change map, the same with the labeled pixels highlighted and the ground truth.

    dataset[_image]_on_modelname_[0.x/r=y/no fine tuning/all]_corrected.png
    # like the previous one, but with spatial correction applied

    dataset[_image]_on_modelname_[0.x/r=y/no fine tuning/all]_heatmap.png
    # Heat map of distances inferred with the network

In addition, there are also files concerning the generation of pseudo-labels:

    dataset_imagename_[ED/SAM]_pseudo_rescaling_[True/False].csv
    # report containing metrics on the pseudo-labels of the "imagename" image in "dataset" with the indicated distance and possible rescaling
    
    dataset_imagename_[ED/SAM]_pseudo_rescaling_[True/False].png
    # image containing the change pseudo-map, the same with the labeled pixels highlighted and the ground truth.
    
    dataset_imagename_[ED/SAM]_pseudo_rescaling_[True/False]_corrected.csv
    # like the previous one, but with spatial correction applied

    dataset_imagename_x%.png
    # maps containing the relative x% of change and non-change pseudo-labels on both single maps and a combined one

    dataset_imagename_radiusy.png
    # maps containing change and non-change pseudo-labels on both single maps and a combined one, extracted by neighborhood with radius y

    

## Install

    pip install -r requirements.txt

**Python  3.8**

Packages:

* [Tensorflow 2.4.1](https://www.tensorflow.org/) 
* [Keras 2.4.3](https://github.com/keras-team/keras)
* [Matplotlib 3.4.1](https://matplotlib.org/)
* [Scipy 1.6.2](https://www.scipy.org/)
* [Numpy 1.19.5](https://www.numpy.org/)
* [Scikit-learn 0.24.1](https://scikit-learn.org/stable/)
* [Scikit-image 0.18.1](https://scikit-image.org/)
* [Hyperas 0.4.1](https://github.com/maxpumperla/hyperas)
* [Hyperopt 0.2.5](https://github.com/hyperopt/hyperopt)
* [Pillow 8.2.0](https://pillow.readthedocs.io/en/stable/)



## Usage

To configure the execution of the algorithm, it is necessary to set the parameters in the *settings* area of net.conf. The meaning of the various fields is given below: 

    [setting]
    train_set         # training dataset name. Said name must be given also to the section with information about it
    test_set          # test dataset name. Said name must be given also to the section with information about it
    distance          # ED=> use euclidean distance, SAM => use SAM. 
    model_name        # name of the model to be saved/loaded
    apply_rescaling   # True=>applies minmax rescaling to data,  False=> Does not apply rescaling
    training          # True => Runs main.py in training mode, False=> runs main.py in testing mode
    fine_tuning       # -1=>doesn't applies fine tuning 0=> applies ft with all the pseudo 1=> applies ft with percentual selection 2=>applies ft with neighborhood selection 
    pseudo_percentage # value in [0,1]. Indicates the pseudo-labels percentage to be used
    pseudo_radius     # positive integer. Indicates the radius for the neighborhood extraction of pseudo-labels

In addition to configuring values for execution, you must also enter information about the dataset you intend to use:

    [nome_dataset]
    imgAPath          # path for the 'before' images
    imgBPath          # path for the 'after' images
    labelPath         # path for the image pair labels
    pseudoPath        # path where pseudo-labels for fine tuning are saved/loaded
    matLabel          # name of the field containing the data (only .mat files)
    changedLabel      # ground truth label indicating a changed pixel pair
    unchangedLabel    # ground truth label indicating a unchanged pixel pair
    unknownLabel      # ground truth label indicating an unknown pixel pair

NB: the datasets should be organized so that the quadruple ("before" image, "after" image, labels, pseudo labels)
are four files (or folders with the images to be composed) with the same name in the paths indicated by the above section.

In addition, it is also possible to change some parameters of the Hyperas automatic search algorithm according to the distance function:

    [hyperas settings distance_abbreviation]
    batch_size      # list (python syntax) of possible batch size values
    max_dropout     # float in ]0,1[ indicating the upper bound of the values that can be chosen for the dropouts
    neurons         # list (python syntax) of possible neuron number to be assigned at the first layer
    neurons_1       # list (python syntax) of possible neuron number to be assigned at the second layer
    neurons_2       # list (python syntax) of possible neuron number to be assigned at the third layer
    fourth_layer    # True=> adds the fourth 512 neuron layer False=>fourth layer isn't added

Once you have set all the necessary values within the configuration file, you can start the code contained in main.py to initiate
the training or prediction phase, the code contained in predutils.py to generate the pseudo-labels or the "labelby"s for map generation.

### Model training

Set the value *training=True*, the values of *train_set* and *test_set* with the names of the respective datasets you intend to use (currently implemented are *"BAY AREA "*, *"SANTA BARBARA "*, *"ONERA TRAIN "* and *"ONERA TEST "*), the value of *distance* indicating the selected distance function (between "*ED*" and "*SAM*"), the value of *model_name=chosen_model* and the value of *apply_rescaling* to *True* (recommended) or *False*. 
Also, change the values of the *hyperas settings SAM/ED* sections as appropriate.
After that you can run main.py without any other arguments

    python3 main.py

This will load and pre-process the dataset, train the network on the indicated sets with hyperparameter optimization and subsequently test it. Specifically, the number of neurons per layer (in the indicated ranges), learning rate, dropout rates and batch size will be optimized. The best model is selected based on the lowest *loss* recorded on the validation set (default=20% of the train set) At the end of the learning a report of the various trials will be saved in *stat* and the resulting best model in *model* (both paths are given in config.py)

### Model testing

Set the value of *training=False*, the value of *test_set* with the name of the dataset to be used for testing, the value of *model_name* with the name of the model in *model* to be tested, and the value of *apply_rescaling*
to *True* (recommended) in case you want to apply rescaling on the input dataset, otherwise to *False*.
In case you want to implement testing with fine tuning, set the value of *fine_tuning* to *0* to use all pseudo-labels, 
*1* to use selection by percentage (and assigning *pseudo_percentage* the desired amount),
*2* to use selection by radius (and assigning *pseudo_radius* the desired radius). Otherwise, to perform testing without tuning, set *fine_tuning=-1*

After that you can run main.py again with the previous command.

This will initiate the loading and pre-processing of the test dataset, the loading of the model, the application of fine tuning (if required), and finally the execution of the prediction on the given test set.
A .csv file containing the testing statistics with and without spatial correction, a heat map representing the distances calculated by the network for the pair of images, and two images containing the predicted maps (with and without correction) will be generated in *stat*. These maps present the predicted map (*Total Prediction*), the same with a mask covering the unknown pixels (*Comparable Prediction*) and the *ground truth*. The color code states that red pixels indicate change, blue pixels indicate no change, and yellow pixels possess no ground truth. 

### Pseudo-labels generation
Set the value of *train_set* with the name of the dataset to be used to calculate the pseudo-labels, the value of *distance* with "*ED*" or "*SAM*" depending on the distance function you want to use for the calculation, the value of *apply_rescaling* to *True* or *False*, in case you want to apply rescaling on the input dataset or not. 
At this point you will be able to run predutils.py without any other arguments
    
    python3 predutils.py
The pseudo labels will be generated by applying the chosen distance on the image pair and deriving the conversion threshold using the Otsu method. They will then be saved in the "pseudoPath" path of the dataset, each with the name of the reference image pair, as a .pickle file. In *stat* also will be collected the maps and statistics for the pseudo-labels, with and without spatial correction (default radius=3).

### Pseudo-labels plotting
In case you want to display the distribution of the labels extracted by the methods by neighborhood or by percentage, you can set the value of *train_set* with the name of the dataset whose pseudo labels are to be printed. Then if you want a plot of the extracted maps by percentage, you can launch labelbyperc_generation.py:

    python3 labelbyperc_generation.py
all maps containing 10% to 90% of the best pseudo-labels will be generated in *stat*, with a 10% difference between each image.
<br>Differently for the plot of maps extracted by radius, set *pseudo_radius* with the maximum radius value to be considered for plotting, and run labelbyneigh_generation.py:

    python3 labelbyneigh_generation.py

All the maps obtained by extraction by neighborhood from radius=2 to the chosen radius will be generated in *stat*.
The format of both images includes one map with only change labels (*C label*), one with only no-change labels (*N label*) and one with both (*NC label*). Yellow indicates the excluded pixels.

#### Pre-processing

During the preprocessing phase, the data contained in the dataset are organized into arrays of triples (pixel A, pixel B, label), normalized to [0, 1] through MinMax. the labels undergo refactoring of the values so that they are always logically consistent with the algorithms used (i.e., changed pixels are labeled with "0," unchanged pixels with "1," and unknown pixels with "2"). In addition, the images are also cleaned of any unlabeled pairs (in case of training).
This phase is implemented before any training or testing phase.


#### Evaluation 

The final evaluation of the performance of individual models is done using a confusion matrix constructed through the real labels of the examples, loaded together with the pairs, and the Siamese Network predictions. Values reported as "true positives" are the correctly predicted changed pixels, "true negatives" are the correctly predicted unchanged pixels, "false positives" are the unchanged pixels predicted as changed, and "false negatives" are the changed pixels predicted as unchanged. In the case of testing, the matrix is constructed both before and after performing the spatial correction (with default radius=3) on the labels predicted by the network, so that *Overall Accuracy* can be calculated at both times with respect to both ground truth and pseudo-labels, where used. Differently for the values calculated on the *validation set* or the *test set during training*, the calculation is performed without the spatial correction.

Another evaluation criterion is the time taken in the execution of the various steps, always recorded in seconds. In the .csv files obtained from the training there is a *time* field indicating the time taken to execute the single *run*.
In the .csv files of the pseudo labels, there are *generation_time* and *correction_time* that indicate the generation and application times of the spatial correction, respectively.
In the .csv files obtained from testing, the fields are present:
    
    prediction_time: time elapsed for the prediction
    extraction_time: time elapsed for the execution of the pseudo-labels extraction algorithms.
                     For the extraction by neighborhood it considers also the time for the spatial correction.  
    correction_time: Time elapsed for the spatial correction on the predicted map.
    ft_time:         time elapsed for the finetuning

#### Required files for script execution

The following files are required to make it possible to run the training or generate the pseudo-labels:

* The dataset containing the images divided into folders between "before," "after," and "labels" in *data/name_dataset/*

Instead, the following files are additionally required for running the testing:
*The model to be tested in *model*, as a pair of files *model_name.h5* and *model_param.pickle*.
* If fine tuning is to be applied, the pseudo-labels as a *imagename.pickle* file in the dedicated pseudo-label folder of the reference dataset.

The pseudo-label files are also needed in case you want to plot them with *"labelby "* scripts.

### Script and main functions
An overview of the main scripts and functions is proposed. For more details, refer to the comments in the code.

#### config.py
This file contains all the constants "internal" to the program, such as the labels used, the margin for loss, the split for *validation*, the paths to the configuration file, the one for saving models, and the one for saving logs. It also contains the variables used to pass datasets and settings to the function that performs the hyperparameter search.

#### dataprocessing.py
This module contains loading and pre-processing functions for the datasets. <br> **load_dataset(name, conf)** allows to load a dataset named "name" (listed in *net.conf*) with the configuration read from the parser "conf". It returns it as a list of "before" images, a list of "after" images, label list and pair name list. Currently, it can load only images in uncompressed ".mat" and ".tif" format (each image counts as a pixel array for a band). To load ".tif" files, they must be placed in folders with the same name of the label files (with extension) and each band must have the name in the correct alphanumeric order (i.e. first band=>B01, second band => B02...). <br> 
**preprocessing(..., conf_section, keep_unlabeled, apply_rescaling)** takes as input the outputs of *load_dataset* (...) and, using the information passed by the parser*conf_section*, applies pre-processing to the dataset. It includes image linearization, MinMaxing on the concatenation of the latter (if *apply_rescaling=True*), changing the labels and generating the two arrays of pixel pairs and respective labels, eventually removing unlabeled pairs (if *keep_unlabeled*=False). The output is a single array containing all the pixel pairs of all the images, together with the respective label array.

#### labelbyneigh_generation.py
Script for generating plots of extracted labels by neighborhood with variable radius. Makes use of the extraction function included in predutils.py

#### labelbyperc_generation.py
Script for generating plots of extracted labels by neighborhood with variable percentage. Makes use of the extraction function included in predutils.py to get the pairs of pixels ordered by distance  (increasing for those changed, decreasing for those not changed) and then generates plots from them.

#### main.py
Main script for performing training or testing of a model. Its use has already been discussed throughout this guide.

#### predutils.py
Module containing the utility functions for prediction as well as the pseudo-label generation script.<br>
**spatial_correction(prediction, radius)** performs spatial correction on the input prediction (already rearranged into a two-dimensional array) by running a kernel of radius *radius* and reassigning to each pixel the predominant class within it. In case of tie, the original class is kept. <br>
**pseudo_labels(first_img, second_img, dist_function, return_distances)** allows pseudo labels to be derived from the pair of images *(first_img, second_img)* with the distance function *dist_function*. Each image has 2 dimensions (height x width, spectral bands). You can decide to get the pseudo label map and the derived threshold directly by setting *return_distances=False*, or get the distance map by setting *return_distances=True*.  <br>
**labels_by_percentage(pseudo_dict, percentage)** extracts from a map of pseudo-labels *pseudo_dict* in dictionary format (see section Pseudo-labels) a percentage indicated by *percentage* (float in ]0, 1]) of the best labels. The measure of goodness considered is the distance between the two pixels: the more extreme it is, the more certain the classification will be. The function, then, sorts the changed pixels in descending order by distance and the unchanged pixels ascending order. After that, a "cut" is performed and two arrays containing the positions of the best percentage pairs and their labels are returned.<br>
**labels_by_neighborhood(pseudo_dict, radius)** extracts from a map of pseudo-labels *pseudo_dict* in dictionary format (see section Pseudo-labels) the best labels. The considered measure is the presence of different pixels in a square neighborhood of radius *radius*. If at least one pixel of different class is present, the middle pixel is discarded. The function then scans the image, removes spurious pixels, and returns two arrays containing the positions of the best pairs and their labels.<br>

#### siamese.py
This module contains all the main functions useful for building, learning and *fine tuning* the Siamese Network model.<br>
**hyperparam_search(train_set, train_labels, test_set, test_labels, distance_function, name, hyperas_search)** allows training with hyperparameter optimization through Hyperas on the indicated train set, with the distance function and settings passed in as input. At each iteration a test is also run on the given set, and at the end of learning the function saves the statistics of the various trials in a .csv file and the resulting best model . It is chosen based on the lowest loss value on the *validation set*. The "name" parameter indicates the name to be given to the saved model.<br>
**siamese_model(train_set, train_labels, test_set, test_labels, score_function)**
 is the one that performs the construction, training and testing of the model for each Hyperas iteration. Returns a dictionary containing the metrics recorded in the current run, according to the Hyperas syntax (Refer to the library documentation for more info). Learning is performed over a maximum of 150 iterates with an *EarlyStopping callback*, i.e., training stops if within a number of epochs (set to 10) the metric being monitored (the loss on the *validation set*) does not exhibit improvement.<br>
**build_net(input_shape, parameters)** builds and compiles a neural network model using the Keras functional API. The function is used both in the training phase and when loading a saved model. As input it requires the *shape* of the single input and a dictionary containing the parameters to be applied for construction. The contents of the dictionary are described in the **Model** section.<br>
**fine_tuning(model, batch_size, x_retrain, pseudo_labels)** takes care of re-training the *model* model, on the dataset and pseudo-labels passed as input. It returns, in addition to the re-trained model, the loss values on *train and valdation set*, the accuracy on *validation*, the number of epochs, and the time taken for the *re-train*.
