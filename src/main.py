import tensorflow as tf

from utils import load_dataset
from model import MCDropoutModel
from train import train
from test import test
from mc_dropout_prediction import mc_dropout_prediction

def main(dump_path, training=True):
    x_train, y_train, x_test, y_test = load_dataset()

    # define each tensor flow dataset and Model
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    model = MCDropoutModel()

    if training:
        # train MC Dropout Model
        epochs = 10
        train(model, train_ds, epochs)

        # test trained model
        test(model, test_ds)

        # save trained model
        model.save_weights(dump_path)
    else:
        # load weights of trained model
        model.load_weights(dump_path)

    # predicted by MC Dropout Model
    mc_dropout_prediction(x_test, y_test, model, sample_num=100, class_num=10)


if __name__ == "__main__":
    dump_path = "../data/model/checkpoints/"
    main(dump_path, training=False)
