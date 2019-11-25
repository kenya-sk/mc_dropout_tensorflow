import tensorflow as tf

from utils import load_dataset
from model import MCDropoutModel
from train import train
from test import test


def main(dump_path):
    x_train, y_train, x_test, y_test = load_dataset()
    # define each tensor flow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # train MC Dropout Model
    epochs = 10
    model = MCDropoutModel()
    train(model, train_ds, epochs)

    # test trained model
    test(model, test_ds)

    # save trained model
    model.load_weights(dump_path)


if __name__ == "__main__":
    dump_path = "../data/checkpoints/mc_dropout_model"
    main(dump_path)
