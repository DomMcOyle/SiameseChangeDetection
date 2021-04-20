import DataProcessing as dp
import Siamese as s

if __name__ == '__main__':
    x_train, y_train = dp.load_aviris_dataset()

    model = s.siamese_model(x_train[0][0].shape)

    model.fit([x_train[:, 0], x_train[:, 1]], y_train,
              batch_size=64,
              epochs=100,
              verbose=2)

    model.save("model")

