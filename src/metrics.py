
from scipy.special import softmax, expit as sigmoid
from sklearn.metrics import f1_score

def ktiv_male_metrics(eval_preds, padding_idx=-100):
    logits, labels = eval_preds
    # shape of logits: (n_samples, n_chars, n_labels=3)
    # shape of labels: (n_samples, n_chars)

    # reshape to 2d to treat each character as a sample: (N, n_labels=15)
    n_labels = logits.shape[-1]
    logits = logits.reshape(-1, n_labels)
    labels = labels.reshape(-1)

    mask = (labels != padding_idx)
    logits = logits[mask]
    labels = labels[mask]

    probs = softmax(logits, axis=-1)
    preds = probs.argmax(axis=-1)

    accuracy = (labels == preds).mean()
    macro_f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

def unikud_metrics(eval_preds, prob_threshold=0.5, padding_idx=-100):
    logits, labels = eval_preds
    # shapes of each: (n_samples, n_chars, n_labels=15)

    # reshape to 2d to treat each character as a sample: (N, n_labels=15)
    n_labels = logits.shape[-1]
    logits = logits.reshape(-1, n_labels)
    labels = labels.reshape(-1, n_labels)

    mask = (labels != padding_idx).all(axis=-1)
    logits = logits[mask]
    labels = labels[mask]

    probs = sigmoid(logits)
    preds = (probs >= prob_threshold).astype(int)

    accuracy = (labels == preds).all(axis=-1).mean()
    macro_f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }