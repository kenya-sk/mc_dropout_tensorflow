import numpy as np

from collections import defaultdict
from tqdm import trange


def mc_dropout_prediction(x_test, y_test, model, sample_num=100, class_num=10):
    pred_dctlst = {"answer": [], "entropy": []}
    for i in trange(len(x_test)):
        image = x_test[i]
        image = image[np.newaxis]
        preds = np.zeros((sample_num, class_num), dtype=np.float32)
        for j in range(sample_num):
            # Enable dropout
            predictions = model(image, training=True)
            preds[j, :] = predictions
        preds = preds.mean(axis=0)
        entropy = np.sum(-preds * np.log(preds))
        pred_dctlst["answer"].append(y_test[i][0])
        pred_dctlst["entropy"].append(entropy)

    return pred_dctlst
