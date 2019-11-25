import tensorflow as tf


@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    # MCDropout enables dropout even during prediction.
    predictions = model(images, training=True)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def test(model, test_ds):
    # define each metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # evaluate by each data
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss.result(),
                          test_accuracy.result() * 100))
