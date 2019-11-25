import tensorflow as tf


def load_dataset():
    #  load cifar 10 dataset
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Train Shape: {0}".format(x_train.shape))
    print("Test Shape: {0}".format(x_test.shape))

    return x_train, y_train, x_test, y_test


def get_label(num):
    num2label = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                    8: "ship", 9: "truck"}

    return num2label[num]
