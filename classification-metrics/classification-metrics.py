import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0
    
    classes = np.unique(np.concatenate([y_true, y_pred]))

    tp_list, fp_list, fn_list, support_list = [], [], [], []
    for cls in classes:
        tp_list.append(np.sum((y_true == cls) & (y_pred == cls)))
        fp_list.append(np.sum((y_true != cls) & (y_pred == cls)))
        fn_list.append(np.sum((y_true == cls) & (y_pred != cls)))
        support_list.append(np.sum(y_true == cls))
        
    tp_arr = np.array(tp_list)
    fp_arr = np.array(fp_list)
    fn_arr = np.array(fn_list)
    support_arr = np.array(support_list)

    if average == "micro":
        total_tp = np.sum(tp_arr)
        total_fp = np.sum(fp_arr)
        total_fn = np.sum(fn_arr)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average in ["macro", "weighted"]:
        precisions = np.where((tp_arr + fp_arr) > 0, tp_arr / (tp_arr + fp_arr), 0.0)
        recalls = np.where((tp_arr + fn_arr) > 0, tp_arr / (tp_arr + fn_arr), 0.0)
        f1s = np.where((precisions + recalls) > 0, 2 * (precisions * recalls) / (precisions + recalls), 0.0)
        
        if average == "macro":
            precision = np.mean(precisions) if len(precisions) > 0 else 0.0
            recall = np.mean(recalls) if len(recalls) > 0 else 0.0
            f1 = np.mean(f1s) if len(f1s) > 0 else 0.0
        else: 
            total_support = np.sum(support_arr)
            if total_support > 0:
                precision = np.sum(precisions * support_arr) / total_support
                recall = np.sum(recalls * support_arr) / total_support
                f1 = np.sum(f1s * support_arr) / total_support
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
                
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }