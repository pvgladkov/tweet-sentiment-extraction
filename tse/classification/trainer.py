from sklearn.metrics import average_precision_score
import numpy as np


class Trainer(object):

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def log_epoch(self, train_loss, eval_loss, acc, epoch):
        self.logger.info('######## epoch={} ########'.format(epoch))
        self.logger.info("train_loss={:.4f}, val_loss={:.4f}, acc={:.4f}".format(train_loss, eval_loss, acc))

    def log_pr(self, labels, predictions, epoch):
        pass

    @staticmethod
    def stack(predictions, labels, step_predictions, step_labels):
        step_predictions = step_predictions.detach().cpu().numpy()[:, 1]
        step_labels = step_labels.detach().cpu().numpy()
        if predictions is None:
            predictions = step_predictions
            labels = step_labels
        else:
            predictions = np.hstack((predictions, step_predictions))
            labels = np.hstack((labels, step_labels))

        return predictions, labels
