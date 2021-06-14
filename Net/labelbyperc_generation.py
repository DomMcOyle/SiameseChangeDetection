import config
import configparser
import pickle
import dataprocessing as dp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

if __name__ == '__main__':

    dataset = "SANTA BARBARA"
    image_name = "barbara"

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    pseudo = open(parser[dataset].get("pseudoPath")+"/" + image_name + ".pickle", "rb")
    dict = pickle.load(pseudo)
    pseudo.close()

    first_img, second_img, labels, names = dp.load_dataset(dataset, parser)
    x_test, y_test = dp.preprocessing(first_img, second_img, labels, parser[dataset], keep_unlabeled=True)
    print(len(x_test))
    print(len(dict['distances']))
    print(dict['shape'])

    distances = dict['distances'].reshape(dict['shape'])
    threshold = dict['threshold']
    # genera indici di etichette
    N = np.where(distances <= threshold)
    C = np.where(distances > threshold)

    # copia valori
    nvalues = distances[N]
    cvalues = distances[C]

    # ottieni matrice di coordinate
    nindexes = np.array(N).transpose()
    cindexes = np.array(C).transpose()

    # associa valori e matrice coordinate e ordina
    nmatrix = np.c_[nvalues, nindexes]
    nmatrix = nmatrix[nmatrix[:, 0].argsort()]

    cmatrix = np.c_[cvalues, cindexes]
    cmatrix = cmatrix[(-cmatrix)[:, 0].argsort()]

    # genera mappe vuote
    nmap = np.full(dict['shape'], config.UNKNOWN_LABEL)
    cmap = np.full(dict['shape'], config.UNKNOWN_LABEL)
    ncmap = np.full(dict['shape'], config.UNKNOWN_LABEL)

    print(cmatrix.shape)
    print(nmatrix.shape)
    jmax = cmatrix.shape[0]/10
    kmax = nmatrix.shape[0]/10
    k = 0
    j = 0
    for i in range(1, 10):
        # copia 10%C
        while j < jmax*i:
            cmap[int(cmatrix[j, 1]), int(cmatrix[j, 2])] = config.CHANGED_LABEL
            ncmap[int(cmatrix[j, 1]), int(cmatrix[j, 2])] = config.CHANGED_LABEL
            j = j + 1
        # copia 10%N
        while k < kmax*i:
            nmap[int(nmatrix[k, 1]), int(nmatrix[k, 2])] = config.UNCHANGED_LABEL
            ncmap[int(nmatrix[k, 1]), int(nmatrix[k, 2])] = config.UNCHANGED_LABEL
            k = k+1
        print(j)
        print(jmax*i)
        print(k)
        print(kmax*i)
        color_map = pltc.ListedColormap(config.COLOR_MAP)
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.imshow(cmap, cmap=color_map, vmin=0, vmax=2)
        ax1.title.set_text("C label")

        ax2.imshow(nmap, cmap=color_map, vmin=0, vmax=2)
        ax2.title.set_text("N label")

        ax3.imshow(ncmap, cmap=color_map, vmin=0, vmax=2)
        ax3.title.set_text("NC label")
        fig.savefig(config.STAT_PATH + dataset + "_" + str(i) + "0%" + ".png",
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
