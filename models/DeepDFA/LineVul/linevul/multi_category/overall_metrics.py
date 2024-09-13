from sklearn.metrics import confusion_matrix
import numpy as np



def compute_tn_fp_fn_tp(y_true, y_pred, labels):
    """
    计算多分类问题中每个类别的TN, FP, FN, TP。
    :param y_true: 真实标签数组
    :param y_pred: 预测标签数组
    :param labels: 所有标签的列表
    :return: 每个类别对应的TN, FP, FN, TP值的字典
    """
    metrics = {}
    for label in labels:
        # 将当前类别视为正类，其余类别视为负类
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)

        # 计算当前类别的混淆矩阵
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        metrics[label] = {'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP}
    return metrics
def overall_metrics(y_trues, y_preds):

    # 示例数据
    # y_trues = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2])
    # y_preds = np.array([0, 2, 1, 1, 0, 0, 2, 0, 2])
    labels = np.unique(y_trues)

    # 计算每个类别的TN, FP, FN, TP
    metrics = compute_tn_fp_fn_tp(y_trues, y_preds, labels)

    TP_total = 0
    FP_total = 0
    FN_total = 0
    for label, values in metrics.items():
        print(f"Class {label}:")
        print(f" TN: {values['TN']}, FP: {values['FP']}, FN: {values['FN']}, TP: {values['TP']}")
        if label in [1,2,3,4,5,6]:
            TP_total = TP_total + values['TP']
            FP_total = FP_total + values['FP']
            FN_total = FN_total + values['FN']




    # 计算精确度（Precision）
    precision = TP_total / (TP_total + FP_total)

    # 计算召回率（Recall）
    recall = TP_total / (TP_total + FN_total)

    # 计算F1得分
    f1 = 2 * (precision * recall) / (precision + recall)

    print("precision:", precision,"recall:", recall, "f1:",f1)

