import tensorflow as tf


@tf.function
def train_step(images, labels, model, optimizer, loss_object, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def train(model, train_ds, epochs):
    # define each metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(images, labels, model, optimizer, loss_object, train_loss, train_accuracy)

        template = "Epoch {}, Loss: {}, Accuracy: {}"
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100))

        # reset each metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
