# Análisis de Cáncer de Mama con Machine Learning

Este proyecto implementa un análisis completo del dataset de cáncer de mama de Wisconsin utilizando diferentes algoritmos de machine learning para clasificar tumores como malignos o benignos.

## Requisitos

Para ejecutar este código necesitas tener instalado Python 3.6 o superior y las siguientes bibliotecas:

```bash
pip install numpy pandas scikit-learn
```

## Uso

Para ejecutar el análisis, simplemente ejecuta el archivo Python:

```bash
python "CANCER 08 04.py"
```

El programa mostrará automáticamente los resultados del análisis en la consola, incluyendo estadísticas descriptivas, métricas de rendimiento de los modelos y conclusiones.

## Estructura del Código

El programa está organizado en las siguientes secciones principales:

### 1. Carga y Exploración de Datos

El programa carga el conjunto de datos de cáncer de mama de Scikit-learn y muestra:
- Dimensiones del dataset
- Distribución de clases (benigno/maligno)
- Estadísticas descriptivas de las características

### 2. Preparación de Datos

- División en conjuntos de entrenamiento (70%) y prueba (30%)
- Normalización de características utilizando StandardScaler

### 3. Entrenamiento y Evaluación de Modelos

Se entrenan y evalúan tres modelos de clasificación:
- Árbol de Decisión
- Random Forest
- Regresión Logística

### 4. Comparación de Modelos

Se comparan los tres modelos en términos de:
- Precisión
- Tiempo de entrenamiento
- Resultados de validación cruzada

### 5. Optimización de Hiperparámetros

- Se selecciona la Regresión Logística como el mejor modelo
- Se realiza una búsqueda de hiperparámetros (GridSearchCV) para la Regresión Logística
- Se evalúa el modelo optimizado y se compara con la versión original

### 6. Conclusiones

Se presentan conclusiones finales sobre:
- Comparación de rendimiento entre modelos
- Mejora obtenida mediante optimización
- Características más importantes para la clasificación
- Recomendaciones para aplicaciones médicas

## Interpretación de Resultados

### Métricas de Evaluación

- **Precisión (Accuracy)**: Proporción de predicciones correctas sobre el total.
- **Precision**: Capacidad del modelo para no etiquetar como positivo un ejemplo negativo.
- **Recall (Sensibilidad)**: Capacidad del modelo para encontrar todos los ejemplos positivos.
- **F1-score**: Media armónica entre precision y recall.
- **Matriz de confusión**: Tabla que muestra los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.

### Interpretación de la Matriz de Confusión

Para un problema de clasificación binaria de cáncer:
- **Verdadero Positivo (VP)**: Tumor maligno correctamente identificado como maligno.
- **Falso Positivo (FP)**: Tumor benigno incorrectamente identificado como maligno.
- **Verdadero Negativo (VN)**: Tumor benigno correctamente identificado como benigno.
- **Falso Negativo (FN)**: Tumor maligno incorrectamente identificado como benigno.

### Importancia de Características

Los modelos basados en árboles (Árbol de Decisión y Random Forest) proporcionan una medida de la importancia de cada característica para la clasificación. Esto puede ayudar a identificar qué características del tumor son más relevantes para determinar si es maligno o benigno.

## Consideraciones Médicas

En aplicaciones médicas como la detección de cáncer, es crucial considerar:

1. **Balance entre precisión e interpretabilidad**: Algunos modelos más precisos pueden ser menos interpretables, lo que puede ser problemático en contextos médicos donde se requiere explicabilidad.

2. **Costo de los errores**: Un falso negativo (no detectar un cáncer existente) generalmente tiene consecuencias más graves que un falso positivo.

3. **Validación clínica**: Los modelos de machine learning deben ser validados en entornos clínicos reales antes de su implementación.

## Ejemplo de Salida

Al ejecutar el programa, verás una salida similar a esta:

```
================================================================================
ANÁLISIS DEL DATASET DE CÁNCER DE MAMA
================================================================================

1. INFORMACIÓN GENERAL DEL DATASET
--------------------------------------------------
Dimensiones del dataset: (569, 31)
Distribución de clases:
  - malignant: 212 (37.26%)
  - benign: 357 (62.74%)

...

7. CONCLUSIONES
================================================================================
1. Comparación de modelos:
   - Árbol de Decisión: Precisión = 0.9123, Tiempo = 0.0156s
   - Random Forest: Precisión = 0.9649, Tiempo = 0.1094s
   - Regresión Logística: Precisión = 0.9532, Tiempo = 0.0312s

2. El mejor modelo fue Regresión Logística con una precisión de 0.9532
   Después de la optimización, la precisión mejoró significativamente
```

## Autor

Este código fue desarrollado como parte del curso Taller de Inteligencia Artificial.
