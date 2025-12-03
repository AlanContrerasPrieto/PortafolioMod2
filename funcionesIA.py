import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def saludar():
    print("Hola, funcionesIA está correctamente importado y listo para usar.")

def plot_time_series(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Genera gráficos de series de tiempo para las variables de un dataset.
    :param df: DataFrame de pandas con los datos de series de tiempo.
    :param feature_cols: Lista de nombres de columnas para las variables 'normales' (features).
    :param target_col: Nombre de la columna para la variable objetivo.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: El DataFrame proporcionado está vacío o no es un DataFrame válido.")
        return

    # --- 1. Gráfico de subplots para las variables 'Normales' (Features) ---

    # Se usa una cuadrícula de 3 filas x 2 columnas para las 6 variables
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    fig.suptitle('Variables de Carga (Features Normales)', fontsize=16, y=1.02)
    
    # El método .flatten() permite iterar sobre los ejes (axes) como una lista simple
    axes = axes.flatten()

    # Aseguramos que tenemos exactamente 6 columnas para graficar en 6 subplots
    if len(feature_cols) == 6:
        for i, col in enumerate(feature_cols):
            # Asegura que la variable es numérica antes de graficar
            if pd.api.types.is_numeric_dtype(df[col]):
                axes[i].plot(df.index, df[col], label=col, color='C{}'.format(i))
                axes[i].set_title(col, fontsize=12)
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].text(0.5, 0.5, f"Columna '{col}' no es numérica.", 
                             horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(col + ' (No Graficable)')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Advertencia: Se esperaban 6 columnas de features para el 2x3, pero se encontraron {len(feature_cols)}. Solo se graficarán las primeras.")
        
        # Gráfica de las que haya, usando el mismo layout
        for i, col in enumerate(feature_cols):
            if i < 6 and pd.api.types.is_numeric_dtype(df[col]):
                axes[i].plot(df.index, df[col], label=col, color='C{}'.format(i))
                axes[i].set_title(col, fontsize=12)
                axes[i].tick_params(axis='x', rotation=45)
            elif i < 6:
                 axes[i].text(0.5, 0.5, f"Columna '{col}' no es numérica.", 
                             horizontalalignment='center', verticalalignment='center')
                 axes[i].set_title(col + ' (No Graficable)')

        # Ocultar subplots vacíos si es necesario
        for j in range(len(feature_cols), 6):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    # --- 2. Gráfico único para la variable Objetivo (Target) ---
    
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df[target_col], color='navy', label=target_col)
        plt.title(f'Variable Objetivo: {target_col}', fontsize=16)
        plt.xlabel('Fecha/Tiempo', fontsize=12)
        plt.ylabel('Valor', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    else:
        print(f"Error: La columna objetivo '{target_col}' no se encuentra en el DataFrame o no es numérica.")

def train_valid_test_split(X: pd.DataFrame,y: pd.Series, test_size: float=0.2, valid_size: float=0.1):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    :param X: DataFrame de características.
    :param y: Serie de etiquetas.
    :param test_size: Proporción de datos para el conjunto de prueba.
    :param valid_size: Proporción de datos para el conjunto de validación.
    :return: Tupla con los conjuntos divididos: (X_train, X_valid, X_test, y_train, y_valid, y_test)."""
    X_train, X_temp, y_train, y_temp= train_test_split(X,y, test_size=test_size, shuffle=False)
    X_valid, X_test, y_valid, y_test    = train_test_split(X_temp,y_temp, test_size=valid_size, shuffle=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def add_time_features(df: pd.DataFrame, datamin: bool=True):
    """
    Agrega características de tiempo al DataFrame basado en el índice de fecha/hora.
    :param df: DataFrame de pandas con un índice de fecha/hora.
    :return: DataFrame con nuevas columnas de características de tiempo. 
    Se utilizan transformaciones seno y coseno para capturar la naturaleza cíclica de las características de tiempo.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex.")

    #df['min'] = df.index.minute
    if datamin:
        min = df.index.minute
        df['min_sin'] = np.sin(2 * np.pi * min / 60)
        df['min_cos'] = np.cos(2 * np.pi * min / 60)

    #df['hour'] = df.index.hour
    hour = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    #df['day'] = df.index.day
    day = df.index.day
    df['day_sin'] = np.sin(2 * np.pi * day / 31)
    df['day_cos'] = np.cos(2 * np.pi * day / 31)

    #df['dayofweek'] = df.index.dayofweek
    dayofweek = df.index.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    #df['month'] = df.index.month
    month = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    #df['dayofyear'] = df.index.dayofyear
    dayofyear = df.index.dayofyear
    df['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 366)
    df['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 366)

    return df

def scale_data(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, 
               y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series):
    """
    Escala las features (X) y la variable objetivo (y) usando MinMaxScaler.
    El escalador se fita SÓLO con los datos de entrenamiento para evitar el 
    'data leakage'.
    
    :return: Datos escalados y los escaladores fiteados.
    """
    # 1. Escalar Features (X)
    scaler_X = MinMaxScaler()
    # Fit y Transform SÓLO en el conjunto de entrenamiento
    X_train_scaled = scaler_X.fit_transform(X_train)
    # Transformar Validacion y Test con el scaler fiteado en Train
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 2. Escalar Target (y)
    # y (variable objetivo) se escala por separado si es una predicción de valor numérico
    scaler_y = MinMaxScaler()
    # y_train debe ser re-formateado a (n_samples, 1) para el scaler
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_valid_scaled = scaler_y.transform(y_valid.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_valid_scaled, X_test_scaled, \
           y_train_scaled, y_valid_scaled, y_test_scaled, \
           scaler_X, scaler_y

def create_sequences(X_data, y_data, sequence_length):
    """
    Crea secuencias (ventanas de tiempo) para el entrenamiento de la LSTM.

    :param X_data: Datos de features escalados (numpy array 2D).
    :param y_data: Datos de target escalados (numpy array 1D).
    :param sequence_length: Longitud de la ventana de tiempo (p. ej., 96 para 4 días si la data es horaria).
    :return: Tuplas de numpy arrays 3D (X_sequences, y_targets).
    """
    X_sequences, y_targets = [], []
    
    # La ventana va desde el inicio hasta el penúltimo punto posible
    for i in range(len(X_data) - sequence_length):
        # 1. Secuencia X (Input): La ventana completa de 'sequence_length' pasos
        # desde la posición i. Incluye todas las features.
        X_seq = X_data[i : (i + sequence_length)]
        X_sequences.append(X_seq)
        
        # 2. Target y (Output): El valor a predecir, que es el valor de 'y'
        # inmediatamente después del final de la secuencia X.
        y_target = y_data[i + sequence_length]
        y_targets.append(y_target)

    # Convertir a numpy arrays 
    return np.array(X_sequences), np.array(y_targets)

def plot_predictions(y_true, y_pred, df_index, title_prefix="Análisis de Predicción en Test"):
    """
    Grafica los valores reales vs. los valores predichos de la variable objetivo.

    :param y_true: Valores reales (desescalados) del conjunto de prueba.
    :param y_pred: Valores predichos (desescalados) del modelo.
    :param df_index: Índice de tiempo (fecha/hora) para el conjunto de prueba.
    :param title_prefix: Prefijo para el título del gráfico.
    """
    start_index = len(df_index) - len(y_true)
    time_index_plot = df_index[start_index:]
    
    # 1. Gráfico 
    plt.figure(figsize=(18, 8))
    
    # Valores Reales (Variable Objetivo)
    plt.plot(time_index_plot, y_true, 
             label='Valores Reales (OT)', 
             color='navy', 
             linestyle='-', 
             linewidth=1.5)
    
    # Predicciones
    plt.plot(time_index_plot, y_pred, 
             label='Predicciones del Modelo (LSTM)', 
             color='red', 
             linestyle='-', 
             linewidth=1.5,
             alpha=0.7)
    
    plt.title(f'{title_prefix} - Valores Reales vs. Predicciones', fontsize=16)
    plt.xlabel('Fecha y Hora', fontsize=12)
    plt.ylabel('Temperatura del Aceite (OT)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_full_series_predictions(df_original: pd.DataFrame, target_col: str, 
                                 y_true: np.ndarray, y_pred: np.ndarray, 
                                 test_start_index: int, sequence_length: int):
    """
    Grafica la serie de tiempo original completa y superpone las predicciones
    del modelo sobre la sección de prueba.

    :param df_original: DataFrame original (completo) con el índice de tiempo.
    :param target_col: Nombre de la columna objetivo (e.g., 'OT').
    :param y_true: Valores reales (desescalados) del conjunto de prueba.
    :param y_pred: Valores predichos (desescalados) del modelo.
    :param test_start_index: El índice (número de fila) donde comienza el 
                             conjunto de prueba en el DataFrame original.
    :param sequence_length: La longitud de la ventana usada en la LSTM (e.g., 96).
    """
    plt.figure(figsize=(20, 8))
    
    # 1. Graficar la Serie Completa Original
    # Se utiliza todo el índice de tiempo y los valores de la columna objetivo
    plt.plot(df_original.index, df_original[target_col], 
             label='Serie Real Completa (OT)', 
             color='gray', 
             linestyle='-', 
             linewidth=1.0, 
             alpha=0.6)
    
    # 2. Determinar la Sección de Predicción
    # El índice de la predicción comienza en el punto donde inicia el test set,
    # más la longitud de la secuencia (porque la primera predicción se hace
    # después de haber visto toda la primera ventana).
    
    # Indice donde comienza la primera predicción
    pred_start_index = test_start_index + sequence_length
    
    # Asegurarse de que el índice de tiempo para la predicción coincida con y_pred
    # Se usan solo los valores del índice que corresponden a las predicciones
    index_pred = df_original.index[pred_start_index : pred_start_index + len(y_pred)]
    
    
    # 3. Graficar las Predicciones
    # Las predicciones se superponen a la serie original.
    plt.plot(index_pred, y_pred, 
             label='Predicciones del Modelo (Test Set)', 
             color='red', 
             linestyle='--', 
             linewidth=2.0)
    
    # Opcional: Graficar el valor real de la sección de prueba para mejor comparación
    plt.plot(index_pred, y_true, 
             label='Valores Reales del Test Set', 
             color='blue', 
             linestyle='-', 
             linewidth=1.5,
             alpha=0.8)
    
    plt.title(f'Predicciones del Modelo LSTM Superpuestas en la Serie Completa de {target_col}', fontsize=16)
    plt.xlabel('Fecha y Hora', fontsize=12)
    plt.ylabel(f'{target_col} (Temperatura Desescalada)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()