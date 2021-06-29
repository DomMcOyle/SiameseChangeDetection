import configparser
import config
import os
import pickle
import numpy as np
import matplotlib.colors as pltc
import matplotlib.pyplot as plt
from predutils import labels_by_neighborhood

if __name__ == '__main__':

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    dataset = parser["settings"].get("train_set")
    radius_max = int(parser["settings"].get("pseudo_radius"))

    for image_name in os.listdir(parser[dataset].get("pseudoPath")):
        print("Info: GENERATING " + os.path.splitext(image_name)[0] + " MAP...")
        pseudo = open(parser[dataset].get("pseudoPath")+"/" + image_name, "rb")
        dict = pickle.load(pseudo)
        pseudo.close()

        for i in range(2, radius_max + 1):
            print("Info: RADIUS " + str(i) + "/" + str(radius_max))
            best_data, labels = labels_by_neighborhood(dict, radius=i)

            nmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)
            cmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)
            ncmap = np.full(dict["shape"][0]*dict["shape"][1], fill_value=2)

            nmap[best_data[np.where(labels == config.UNCHANGED_LABEL)]] = config.UNCHANGED_LABEL
            cmap[best_data[np.where(labels == config.CHANGED_LABEL)]] = config.CHANGED_LABEL
            ncmap[best_data] = labels

            nmap = np.reshape(nmap, dict["shape"])
            cmap = np.reshape(cmap, dict["shape"])
            ncmap = np.reshape(ncmap, dict["shape"])

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

            fig.savefig(config.STAT_PATH + dataset + "_" + os.path.splitext(image_name)[0] + "_radius" + str(i) + ".png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
