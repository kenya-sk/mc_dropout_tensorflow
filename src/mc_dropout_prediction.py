import numpy as np

from collections import defaultdict
from tqdm import trange


def mc_dropout_prediction(model, x_test, y_test, sample_num=50, class_num=10):
    entropy_dctlst = defaultdict(list)
    for i in trange(len(x_test)):
        image = x_test[i]
        image = image[np.newaxis]
        preds = np.zeros((sample_num, class_num), dtype=np.float32)
        for j in range(sample_num):
            predictions = model(image)
            preds[j, :] = predictions
        preds = preds.mean(axis=0)
        entropy = np.sum(-preds * np.log(preds))
        entropy_dctlst[y_test[i][0]].append(entropy)

    return entropy_dctlst
