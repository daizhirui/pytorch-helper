import numpy as np


def multiclass_score_to_accuracy(score, label):
    pred = np.argmax(score, axis=1)
    acc = np.sum(pred == label) / pred.size
    return acc


def confusion_matrix(pred, gt, num_classes):
    # row: ground truth
    # column: prediction
    coding = gt * num_classes + pred
    counting = np.bincount(coding, minlength=num_classes * num_classes)
    return counting.reshape(num_classes, num_classes)


def confusion_matrix_to_accuracy(conf):
    # accuracy: (TP + TN) / (TP + TN + FP + FN)
    return np.diag(conf).sum() / (np.sum(conf) + 1.e-16)


def confusion_matrix_to_precision(conf):
    # precision: TP / (TP + FP)
    return np.diag(conf) / (np.sum(conf, axis=1) + 1.e-16)


def confusion_matrix_to_recall(conf):
    # recall: TP / (TP + FN)
    return np.diag(conf) / (np.sum(conf, axis=0) + 1.e-16)


def f1(recall, precision):
    return 2 * recall * precision / (recall + precision + 1.e-16)


class ConfusionMatrix:
    def __init__(self, pred, gt, num_classes):
        self.matrix = confusion_matrix(pred, gt, num_classes)

    def accuracy(self) -> float:
        return confusion_matrix_to_accuracy(self.matrix)

    def precision(self) -> np.ndarray:
        return confusion_matrix_to_precision(self.matrix)

    def recall(self) -> np.ndarray:
        return confusion_matrix_to_recall(self.matrix)

    def f1(self) -> float:
        return f1(self.recall(), self.precision())
