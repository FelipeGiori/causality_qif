from sklearn.metrics import roc_auc_score


def forward_auc(y_true, y_pred):
    target_one = [1 if x == 1 else 0 for x in y_true]
    score = roc_auc_score(target_one, y_pred)
    return score


def reverse_auc(y_true, y_pred):
    target_neg_one = [1 if x == -1 else 0 for x in y_true]
    neg_predictions = [-x for x in y_pred]
    score = roc_auc_score(target_neg_one, neg_predictions)
    return score


def bidirectional_auc(y_true, y_pred):
    score_forward = forward_auc(y_true, y_pred)
    score_reverse = reverse_auc(y_true, y_pred)
    score = (score_forward + score_reverse) / 2.0
    return score