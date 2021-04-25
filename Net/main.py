import dataprocessing as dp
import siamese as s
import keras.models as km
import os
import numpy as np

if __name__ == '__main__':
    """
    nuovo workflow per il data
    - creare il configparser
    - caricare il dataset
    - eseguire il preprocessing
    
    """
    """
    x_train, y_train = dp.load_aviris_dataset("sb")

    model = s.siamese_model(x_train[0][0].shape)

    model.fit([x_train[:, 0], x_train[:, 1]], y_train,
              batch_size=64,
              epochs=10,
              verbose=2)
    model.save_weights("model"+os.sep + "weights4.h5")
#   model.save("model")
    """
    x_test, y_test = dp.load_aviris_dataset("sb")
    trained_model = s.siamese_model(x_test[0][0].shape)
    trained_model.load_weights("model"+os.sep + "weights4.h5")
    prediction = trained_model.predict([x_test[0, 0], x_test[0, 1]])
    print(np.unique(prediction))