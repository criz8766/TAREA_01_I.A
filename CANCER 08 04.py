import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
# En la parte inicial, modifica la configuración de warnings
import warnings
# Filtrar las advertencias específicas de convergencia
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="The max_iter was reached which means the coef_ did not converge")

# 1. Cargar el dataset de cáncer de mama
print("=" * 80)
print("ANÁLISIS DEL DATASET DE CÁNCER DE MAMA")
print("=" * 80)

# Cargar el dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Convertir a DataFrame para mejor manipulación
df = pd.DataFrame(X, columns=feature_names)
df['diagnosis'] = y

print("\n1. INFORMACIÓN GENERAL DEL DATASET")
print("-" * 50)
print(f"Dimensiones del dataset: {df.shape}")
print(f"Distribución de clases:")
for i, name in enumerate(target_names):
    count = np.sum(df['diagnosis'] == i)
    percentage = count / len(df) * 100
    print(f"  - {name}: {count} ({percentage:.2f}%)")

# 2. Exploración básica de datos
print("\n2. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 50)
print(df.describe().T)

# 3. Preparación de datos
print("\n3. PREPARACIÓN DE DATOS")
print("-" * 50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Normalización de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Datos normalizados con StandardScaler")

# 4. Entrenamiento y evaluación de modelos
print("\n4. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("=" * 80)

models = {
    'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Regresión Logística': LogisticRegression(max_iter=5000, random_state=42)  # Aumentado de 1000 a 5000
}

results = {}
training_times = {}
cv_scores = {}

for name, model in models.items():
    print(f"\n{name.upper()}")
    print("-" * 50)
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    training_times[name] = training_time
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Validación cruzada
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = cv_score
    
    # Guardar resultados
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'conf_matrix': conf_matrix
    }
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.4f} segundos")
    print(f"Validación cruzada (5-fold): {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    
    print("\nInforme de clasificación:")
    print(report)
    
    print("\nMatriz de confusión:")
    print("Filas: Etiqueta Real, Columnas: Etiqueta Predicha")
    for i, target in enumerate(target_names):
        print(f"  {target}: {conf_matrix[i]}")
    
    # Importancia de características (para modelos basados en árboles)
    if hasattr(model, 'feature_importances_'):
        print("\nImportancia de características (top 10):")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 5. Comparación de modelos
print("\n5. COMPARACIÓN DE MODELOS")
print("=" * 80)

print("\nComparación de precisión:")
for name in models.keys():
    print(f"  - {name}: {results[name]['accuracy']:.4f}")

print("\nComparación de tiempos de entrenamiento:")
for name in models.keys():
    print(f"  - {name}: {training_times[name]:.4f} segundos")

print("\nComparación de validación cruzada:")
for name in models.keys():
    print(f"  - {name}: {cv_scores[name].mean():.4f} ± {cv_scores[name].std():.4f}")

# 6. Optimización de hiperparámetros para el mejor modelo
print("\n6. OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("=" * 80)

# Determinar el mejor modelo basado en precisión
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"El mejor modelo es: {best_model_name}")

# Configurar búsqueda de hiperparámetros para el mejor modelo
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Árbol de Decisión':
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    best_model = DecisionTreeClassifier(random_state=42)
else:  # Regresión Logística
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga']
    }
    best_model = LogisticRegression(max_iter=5000, random_state=42)  # Aumentado de 1000 a 5000

print("\nBuscando mejores hiperparámetros...")
print(f"Espacio de búsqueda: {param_grid}")

# Realizar búsqueda de hiperparámetros
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores hiperparámetros para {best_model_name}:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"Mejor precisión con validación cruzada: {grid_search.best_score_:.4f}")

# Evaluar el modelo optimizado
optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test_scaled)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
report_optimized = classification_report(y_test, y_pred_optimized, target_names=target_names)

print(f"\nPrecisión del modelo optimizado: {accuracy_optimized:.4f}")
print("Informe de clasificación del modelo optimizado:")
print(report_optimized)

print(f"\nComparación de precisión: Original vs Optimizado")
print(f"  - Original: {results[best_model_name]['accuracy']:.4f}")
print(f"  - Optimizado: {accuracy_optimized:.4f}")
print(f"  - Mejora: {(accuracy_optimized - results[best_model_name]['accuracy']) * 100:.2f}%")

# 7. Conclusiones
print("\n7. CONCLUSIONES")
print("=" * 80)
print("1. Comparación de modelos:")
for name in models.keys():
    print(f"   - {name}: Precisión = {results[name]['accuracy']:.4f}, Tiempo = {training_times[name]:.4f}s")

print(f"\n2. El mejor modelo fue {best_model_name} con una precisión de {results[best_model_name]['accuracy']:.4f}")
print(f"   Después de la optimización, la precisión mejoró a {accuracy_optimized:.4f}")

print("\n3. Características más importantes (para modelos basados en árboles):")
if 'Random Forest' in models and hasattr(models['Random Forest'], 'feature_importances_'):
    rf_importances = models['Random Forest'].feature_importances_
    top_features_rf = [feature_names[i] for i in np.argsort(rf_importances)[-5:]]
    print(f"   - Random Forest: {', '.join(top_features_rf)}")
if 'Árbol de Decisión' in models and hasattr(models['Árbol de Decisión'], 'feature_importances_'):
    dt_importances = models['Árbol de Decisión'].feature_importances_
    top_features_dt = [feature_names[i] for i in np.argsort(dt_importances)[-5:]]
    print(f"   - Árbol de Decisión: {', '.join(top_features_dt)}")

print("\n4. Recomendaciones:")
print(f"   - Para este problema de clasificación de cáncer de mama, se recomienda utilizar {best_model_name}")
print(f"   - Los hiperparámetros óptimos son: {grid_search.best_params_}")
print("   - Es importante normalizar los datos debido a las diferentes escalas de las características")
print("   - La interpretabilidad del modelo es crucial en aplicaciones médicas, por lo que se debe")
print("     considerar el equilibrio entre precisión y capacidad de explicación")