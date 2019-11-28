from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, GlobalAveragePooling2Ds
from tensorflow.keras import Model

class MCDropoutModel(Model):
    def __init__(self):
        super(MCDropoutModel, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), activation="relu")
        self.dropout1 = Dropout(0.5)
        self.pool1 = MaxPool2D()
        # second layer
        self.conv2 = Conv2D(128, (3, 3), activation="relu")
        self.dropout2 = Dropout(0.5)
        self.pool2 = MaxPool2D()
        # third layer
        self.conv3 = Conv2D(256, (3, 3), activation="relu")
        self.dropout3 = Dropout(0.5)
        self.pool3 = GlobalAveragePooling2D()
        self.flatten = Flatten()
        # fourth layer
        self.dense1 = Dense(1024, activation="relu")
        self.dropout4 = Dropout(0.5)
        # output
        self.dense2 = Dense(10, activation="softmax")

    def call(self, x, training):
        x = self.conv1(x)
        x = self.dropout1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout3(x, training=training)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x, training=training)
        y = self.dense2(x)

        return y
