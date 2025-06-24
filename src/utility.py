"""
flowchart LR
    utility.py -->  trn.py
    utility.py -->  tst.py

    trn.py -->      |beta.csv| results
    tst.py -->      |beta.csv| results
    tst.py -->      |cmatriz.csv, fscores.csv| results

    data[dtrain.csv, dtest.csv]
    config[config.csv]

    data --> trn.py
    config --> trn.py
    trn.py --> results/beta.csv
    data --> tst.py
    config --> tst.py
    results/beta.csv --> tst.py
"""

import numpy as np
import pandas as pd

def load_data_csv(data_path, config_path):
    """
    Carga los datos y parámetros de configuración.

    Parámetros:
    - data_path: ruta al archivo CSV de datos (última columna de etiquetas).
    - config_path: ruta al CSV de configuración (sigma2 y lambda).

    Retorna:
    - X: matriz de características (numpy.ndarray).
    - y: vector de etiquetas (numpy.ndarray).
    - sigma2: varianza del kernel Gaussiano (float).
    - lambd: parámetro de regularización (float).
    """
    # Carga de datos
    df = pd.read_csv(data_path)
    *feat_cols, label_col = df.columns
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    # Carga de configuración (dos filas: sigma2, lambda)
    conf = pd.read_csv(config_path, header=None).values.flatten()
    sigma2 = float(conf[0])
    lambd = float(conf[1])

    return X, y, sigma2, lambd

def kernel_mtx(X1, X2, sigma2):
    """
    Construye la matriz de kernel Gaussiano (RBF) entre dos conjuntos.

    Parámetros:
    - X1: matriz n1 x d.
    - X2: matriz n2 x d.
    - sigma2: varianza del kernel.

    Retorna:
    - K: n1 x n2.
    """
    # distancias al cuadrado
    diff = X1[:, None, :] - X2[None, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    K = np.exp(-sq_dists / (2 * sigma2))
    return K

def krr_coeff(K, y, lambd):
    """
    Calcula coeficientes beta para Kernel Ridge Regression.

    Parámetros:
    - K: kernel n x n.
    - y: etiquetas (n,).
    - lambd: regularización.

    Retorna:
    - beta (n,).
    """
    n = K.shape[0]
    K_reg = K + lambd * np.eye(n)
    beta = np.linalg.solve(K_reg, y)
    return beta

def confusion_mtx(y_true, y_pred):
    """
    Calcula matriz de confusión sin usar sklearn.

    Parámetros:
    - y_true: vector de verdaderos.
    - y_pred: vector de predichos.

    Retorna:
    - cm_df: DataFrame con filas=verdaderos, cols=predichos.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((labels.size, labels.size), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df

def metricas(y_true, y_pred):
    """
    Calcula precisión, recall y F1-score por clase manualmente.

    Parámetros:
    - y_true: vector de verdaderos.
    - y_pred: vector de predichos.

    Retorna:
    - metrics_df: DataFrame con columnas ['precision','recall','f1'] e índice=etiquetas.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    # inicializar
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        # precisión y recall seguros
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    metrics_df = pd.DataFrame({
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    }, index=labels)
    return metrics_df
