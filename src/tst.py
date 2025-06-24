import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import load_data_csv, kernel_mtx, confusion_mtx, metricas

# -----------------------------------------------------------------------------
# Script de prueba para Kernel Ridge Regression
# -----------------------------------------------------------------------------
# Este script carga los datos de prueba y parámetros de configuración,
# lee los coeficientes beta entrenados, calcula la matriz de kernel entre
# test y train, genera predicciones, evalúa matriz de confusión y F-scores,
# y guarda los resultados en archivos CSV.
# -----------------------------------------------------------------------------

def main():
    # Definir rutas de archivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '..', 'config', 'config.csv')
    train_data_path = os.path.join(base_dir, '..', 'data', 'dtrain.csv')
    test_data_path = os.path.join(base_dir, '..', 'data', 'dtest.csv')
    beta_path = os.path.join(base_dir, '..', 'results', 'beta.csv')
    cmat_path = os.path.join(base_dir, '..', 'results', 'cmatriz.csv')
    fscores_path = os.path.join(base_dir, '..', 'results', 'fscores.csv')

    # Cargar datos de entrenamiento para obtener X_train y parámetros
    X_train, _, sigma2, _ = load_data_csv(train_data_path, config_path)

    # Cargar datos de prueba y etiquetas reales
    X_test, y_true, _, _ = load_data_csv(test_data_path, config_path)

    # Leer coeficientes beta
    beta = pd.read_csv(beta_path)['beta'].values

    # Calcular matriz de kernel test vs train
    K_test = kernel_mtx(X_test, X_train, sigma2)

    # Generar predicciones continuas y luego binarizarlas (signo)
    y_cont = K_test.dot(beta)
    y_pred = np.where(y_cont >= 0,  1, -1)

    # Evaluar matriz de confusión
    cm_df = confusion_mtx(y_true, y_pred)
    cm_df.to_csv(cmat_path, index=True)
    
    labels = cm_df.index.tolist()
    cm = cm_df.values

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Real Value')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    heatmap_path = os.path.join(base_dir, '..', 'results', 'cmatriz_heatmap.png')
    fig.savefig(heatmap_path)
    plt.close(fig)

    # Evaluar métricas: precision, recall, f1
    metrics_df = metricas(y_true, y_pred)
    metrics_df.to_csv(fscores_path, index=True)

    # Guardar sólo vector de F1 en fscores.csv (2x1, sin header)
    f1_df = metrics_df[['f1']].copy()
    f1_df.to_csv(fscores_path, index=False, header=False)

    print(f"Matriz de confusión guardada en: {cmat_path}")
    print(f"F-scores guardados en: {fscores_path}")

    # Mostrar F-scores en consola en porcentaje
    labels = cm_df.index.tolist()
    f1_pct = (metrics_df['f1'] * 100).round(2)
    print(f"F-score (%):  Clase#{labels[0]} = {f1_pct.iloc[0]:.2f}   Clase#{labels[1]} = {f1_pct.iloc[1]:.2f}")


if __name__ == '__main__':
    main()
