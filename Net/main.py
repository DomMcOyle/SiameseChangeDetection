import DataProcessing as dp

if __name__ == '__main__':
    """
    3.fittare il modello 
    4.salvarlo
    5.provare a predire
    """
    x_train, y_train, x_test, y_test = dp.load_aviris_dataset()

    