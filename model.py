import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models.knn_model import KNN

CSV_PATH = 'dataset_lsm.csv'

def preparar_datos(csv_path, test_size=0.2, random_state=0):
    print('Cargando datos')
    df = pd.read_csv(csv_path)

    print(f'Datos cargados: {len(df)} registros')
    print(f"Clases: {df['clase'].unique()}")

    # Extraer características de pixeles y asegurar que sean numéricas
    X_pixels = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # Procesar secuencias de dedos
    secuencias_left, secuencias_right = [], []

    for idx in range(len(df)):
        # Secuencia izquierda
        try:
            sec_left = df.iloc[idx]['secuencia_dedos_centrales_left']
            if isinstance(sec_left, str):
                sec_left = ast.literal_eval(sec_left)
            sec_array = np.array(sec_left).flatten()
            # Asegurar que todos los valores sean numéricos
            sec_array = pd.to_numeric(sec_array, errors='coerce').fillna(0)
            secuencias_left.append(sec_array)
        except Exception as e:
            print(f"Error procesando secuencia izquierda índice {idx}: {e}")
            secuencias_left.append(np.zeros(25))

        # Secuencia derecha
        try:
            sec_right = df.iloc[idx]['secuencia_dedos_centrales_right']
            if isinstance(sec_right, str):
                sec_right = ast.literal_eval(sec_right)
            sec_array = np.array(sec_right).flatten()
            # Asegurar que todos los valores sean numéricos
            sec_array = pd.to_numeric(sec_array, errors='coerce').fillna(0)
            secuencias_right.append(sec_array)
        except Exception as e:
            print(f"Error procesando secuencia derecha índice {idx}: {e}")
            secuencias_right.append(np.zeros(25))

    # Convertir a arrays numpy y asegurar tipo float
    secuencias_left = np.array(secuencias_left, dtype=float)
    secuencias_right = np.array(secuencias_right, dtype=float)

    # Combinar todas las características
    X_combined = np.hstack([secuencias_left, secuencias_right, X_pixels])
    
    # Verificar que no haya valores no numéricos
    print(f"Tipos de datos en X_combined: {X_combined.dtype}")
    print(f"¿Hay NaN?: {np.any(np.isnan(X_combined))}")
    print(f"¿Hay inf?: {np.any(np.isinf(X_combined))}")

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    y = df['clase'].values

    print(f'Características totales: {X_combined.shape}')
    print(f'Etiquetas: {y.shape}')

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
    )

    print(f'Datos de entrenamiento: {X_train.shape[0]} registros')
    print(f'Datos de prueba: {X_test.shape[0]} registros')

    return X_train, X_test, y_train, y_test, scaler

def plot_confusion_matrix(y_true, y_pred, classes, title="Matriz de Confusión", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Guardar la imagen si se proporciona un path
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()

def entrenar_y_guardar():
    # Preparar datos con división train/test
    X_train, X_test, y_train, y_test, scaler = preparar_datos(CSV_PATH)
    
    i = 5
    model = KNN(k=i)
    
    # Entrenar solo con datos de entrenamiento
    model.fit(X_train, y_train)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    modelo = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'classes': np.unique(y_train).tolist(),
        'y_train': y_train,
        'k': i,
        'modelo_nombre': 'MiKNN'
    }

    # Predecir con datos de prueba (no con los de entrenamiento)
    y_pred = model.predict(X_test)

    print("=== RESULTADOS EN DATOS DE PRUEBA ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # También podemos evaluar en entrenamiento para comparar
    y_pred_train = model.predict(X_train)
    print("=== RESULTADOS EN DATOS DE ENTRENAMIENTO ===")
    print(f"Accuracy (train): {accuracy_score(y_train, y_pred_train):.4f}")
    
    # Matriz de confusión con datos de prueba
    plot_confusion_matrix(y_test, y_pred, classes=np.unique(y_train), 
                         title="Matriz de Confusión - Datos de Prueba")

    joblib.dump(modelo, 'modelo.pkl')
    print('Modelo guardado')

    return model

if __name__ == '__main__':
    entrenar_y_guardar()