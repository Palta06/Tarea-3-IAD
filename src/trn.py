import os
import numpy as np
import pandas as pd
from utility import load_data_csv, kernel_mtx, krr_coeff

# -----------------------------------------------------------------------------
# Script de entrenamiento para Kernel Ridge Regression
# -----------------------------------------------------------------------------
# Este script carga los datos de entrenamiento y parámetros de configuración,
# calcula la matriz de kernel Gaussiano, estima los coeficientes beta y
# guarda el vector beta en un archivo CSV.
# -----------------------------------------------------------------------------

def main():
    # Definir rutas de archivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'dtrain.csv')
    config_path = os.path.join(base_dir, '..', 'config', 'config.csv')
    output_path = os.path.join(base_dir, '..', 'results', 'beta.csv')

    # Cargar datos y parámetros
    X, y, sigma2, lambd = load_data_csv(data_path, config_path)

    # Construir matriz de kernel Gaussiano
    K = kernel_mtx(X, X, sigma2)

    # Calcular coeficientes beta
    beta = krr_coeff(K, y, lambd)

    # Guardar beta en CSV
    df_beta = pd.DataFrame(beta, columns=['beta'])
    df_beta.to_csv(output_path, index=False)
    print(f"Coeficientes beta guardados en: {output_path}")

if __name__ == '__main__':
    main()
