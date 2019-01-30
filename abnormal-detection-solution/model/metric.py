import keras.backend as K
import numpy as np

def F1():
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # abnormal to abnormal TP
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))      # all predicted abnormal samples (TP + FP)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # abnormal to abnormal TP
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # all true abnormal samples (TP + FN)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def f1(y_true, y_pred):
        P = precision(y_true, y_pred)
        R = recall(y_true, y_pred)       
        F = 2 * P * R / (P + R + K.epsilon())  # 2*p*r/(p + r)
        return F

    return f1

def R0():
    def recall_0(y_true, y_pred):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_add = y_true + y_pred #0, 1, 2
        true_negatives = K.sum(K.cast(K.less(y_add, 1), 'float32'))
        all_negatives = K.sum(K.cast(K.less(y_true, 1), 'float32'))
        recall = true_negatives / (all_negatives + K.epsilon())
        return recall
    
    return recall_0

def R1():
    def recall_1(y_true, y_pred):
        true_positives = K.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), 'float32')  # abnormal to abnormal TP
        all_positives = K.cast(K.sum(K.round(K.clip(y_true, 0, 1))), 'float32')  # all true abnormal samples (TP + FN)
        recall = true_positives / (all_positives + K.epsilon())
        return recall
    
    return recall_1

def my_accuracy():
    def my_acc(y_true, y_pred):
        y_pred = K.switch(y_pred < 0, tf.constant(-1), tf.constant(1))
        same = K.cast(K.equal(y_true, y_pred), 'int8')
        acc = K.mean(K.cast(same, 'float32'), axis=-1)
        return acc

    return my_acc


def any_accuracy():
    def any_acc(y_true, y_pred):
        y_pred = K.any(K.cast(K.round(y_pred), 'bool'), axis=1)
        y_true = K.any(K.cast(y_true, 'bool'), axis=1)
        same = K.cast(K.equal(y_true, y_pred), 'int8')
        acc = K.mean(K.cast(same, 'float32'), axis=-1)
        return acc

    return any_acc


def all_accuracy():
    def all_acc(y_true, y_pred):
        same = K.cast(K.equal(y_true, K.round(y_pred)), 'int8')  # convert boolean to integer
        acc_batch = K.min(same, axis=-1, keepdims=False)  # the minimum 0(unequal) or 1(equal)
        acc = K.mean(K.cast(acc_batch, 'float32'), axis=-1)  # convert boolean to float
        return acc
        
    return all_acc



