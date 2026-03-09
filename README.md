```markdown
# 📉 Telecom X — Predicción de Cancelación de Clientes (Churn)

Proyecto de **Ciencia de Datos y Machine Learning** enfocado en la detección temprana de clientes con alta probabilidad de abandonar los servicios de una empresa de telecomunicaciones.

El objetivo principal es **identificar patrones de comportamiento que permitan predecir el churn**, facilitando la implementación de estrategias de retención de clientes.

---

# 🎯 Objetivo del Proyecto

El churn de clientes representa uno de los mayores problemas en empresas de telecomunicaciones. Este proyecto desarrolla un **modelo predictivo basado en Random Forest** capaz de identificar clientes con riesgo de cancelación a partir de variables relacionadas con:

- servicios contratados
- facturación
- comportamiento de uso
- antigüedad del cliente
- métodos de pago

Esto permite a la empresa:

- anticipar la pérdida de clientes
- optimizar campañas de retención
- reducir costos de adquisición de nuevos clientes

---

# 🧠 Enfoque de Ciencia de Datos

El proyecto sigue un flujo completo de **Machine Learning aplicado**:

1️⃣ Preparación y limpieza de datos  
2️⃣ Análisis exploratorio  
3️⃣ Identificación de desequilibrio de clases  
4️⃣ Transformación automática de variables  
5️⃣ Selección de variables relevantes  
6️⃣ Entrenamiento de modelos  
7️⃣ Optimización mediante Pipeline  
8️⃣ Balanceo de datos con SMOTE  
9️⃣ Validación cruzada  
🔟 Exportación del modelo final

---

# 📂 Estructura del Proyecto

```

telecom-churn-prediction/

│
├── Telecom_X_prediccion_cancelacion_churn.ipynb
│   Notebook principal con todo el análisis
│
├── Analisis_Telecom_X_Limpio_y_Transformado
│   Dataset limpio utilizado para el entrenamiento
│
├── modelo_churn_random_forest_pipeline.pkl
│   Modelo final exportado
│
├── Requeriments.txt
│   Librerías necesarias para reproducir el proyecto
│
└── README.md

````

---

# 📊 Dataset

El dataset contiene información de **más de 7000 clientes** de telecomunicaciones.

### Variables principales

| Variable | Descripción |
|------|------|
| gender | Género del cliente |
| SeniorCitizen | Si el cliente es adulto mayor |
| Partner | Si tiene pareja |
| Dependents | Si tiene dependientes |
| tenure | Antigüedad del cliente |
| InternetService | Tipo de servicio de internet |
| Contract | Tipo de contrato |
| PaymentMethod | Método de pago |
| ChargesMonthly | Cargo mensual |
| ChargesTotal | Cargo total acumulado |
| ChargesDaily_Total | Cargo diario calculado |
| Churn | Variable objetivo (canceló o no) |

---

# ⚙️ Instalación del Proyecto

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/tuusuario/telecom-churn-prediction.git
cd telecom-churn-prediction
````

---

### 2️⃣ Crear entorno virtual

```bash
python -m venv venv
```

Activar entorno:

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

---

### 3️⃣ Instalar dependencias

Las librerías necesarias son:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* statsmodels
* imbalanced-learn
* joblib
* IPython

Todas se encuentran listadas en el archivo de dependencias. 

Instalar:

```bash
pip install -r Requeriments.txt
```

---

# 🔎 Proceso de Preparación de Datos

Durante el proceso de preparación se realizaron las siguientes transformaciones:

### Limpieza de identificadores

Se eliminó el identificador del cliente (`customerID`) para evitar que el modelo aprenda patrones irrelevantes.

---

### Consolidación de variables de telefonía

Se agruparon variables relacionadas con el servicio telefónico para mejorar la representación de la información.

---

# ⚖️ Problema de Desequilibrio de Clases

El dataset presenta **desbalance en la variable objetivo (Churn)**.

Para resolver esto se utilizó:

```
SMOTE (Synthetic Minority Over-sampling Technique)
```

Esto genera muestras sintéticas de la clase minoritaria para equilibrar el entrenamiento del modelo.

---

# 🤖 Modelado Predictivo

Se evaluaron diferentes enfoques:

* DummyClasisifier (baseline)
* Árbol de decisión 
* Optimización de profundidad del árbol
* Pipeline completo de Machine Learning

El modelo final seleccionado fue:

### 🌲 Random Forest Classifier

Debido a su capacidad para:

* manejar variables categóricas transformadas
* reducir overfitting
* capturar relaciones no lineales

---

# ⚙️ Pipeline de Machine Learning

El modelo final se construyó utilizando un **Pipeline de Scikit-Learn**, que integra:

```
Pipeline
│
├── ColumnTransformer
│       ├── OneHotEncoder
│       └── StandardScaler
│
└── RandomForestClassifier
```

Esto permite que el modelo pueda recibir **datos en su formato original** y aplicar automáticamente todas las transformaciones necesarias.

---

# 📦 Modelo Exportado

El modelo final fue exportado utilizando `joblib`.

```
modelo_churn_random_forest_pipeline.pkl
```

Este archivo contiene:

* el **transformador de variables ya entrenado**
* el **modelo Random Forest ya entrenado**

El pipeline integra ambos componentes, permitiendo reutilizar el modelo directamente sobre nuevos datos.

---

# ⚠️ Importante sobre el uso del modelo

El modelo espera recibir **las columnas originales del dataset**, no las columnas transformadas.

Ejemplo correcto de entrada:

```
InternetService
Contract
PaymentMethod
gender
Phone_Category
tenure
ChargesMonthly
ChargesTotal
ChargesDaily_Total
```

El pipeline se encargará automáticamente de:

```
datos nuevos
   ↓
ColumnTransformer
   ↓
RandomForest
   ↓
predicción
```

---

# 🚀 Cómo usar el modelo entrenado

### Cargar el modelo

```python
import joblib

modelo = joblib.load("modelo_churn_random_forest_pipeline.pkl")
```

---

### Crear datos de prueba

```python
import pandas as pd

nuevo_cliente = pd.DataFrame([{
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "Phone_Category": "Single Line",
    "tenure": 5,
    "ChargesMonthly": 75.5,
    "ChargesTotal": 300,
    "ChargesDaily_Total": 2.5
}])
```

---

### Realizar predicción

```python
prediccion = modelo.predict(nuevo_cliente)

print(prediccion)
```

Resultado:

```
0 → cliente se mantiene
1 → cliente con riesgo de churn
```

---

# 📈 Visualizaciones del Análisis

El notebook incluye múltiples visualizaciones importantes:

### Distribución de Churn

* proporción de cancelaciones
* análisis del desbalance de clases

### Correlación de variables

Mapa de calor para identificar variables con mayor impacto en churn.

### Evaluación del modelo

* matriz de confusión
* métricas de clasificación
* análisis de desempeño

Todas estas visualizaciones se encuentran dentro del notebook principal.

---

# 📊 Métricas de Evaluación

El modelo fue evaluado utilizando:

* Accuracy
* Precision
* Recall
* F1 Score
* Validación cruzada

Esto garantiza que el modelo tenga **capacidad de generalización sobre nuevos datos**.

---

# 🎓 Conclusión

El proyecto demuestra cómo aplicar un flujo completo de **Machine Learning para la predicción de churn**, integrando:

* preparación de datos
* ingeniería de características
* balanceo de clases
* pipeline de procesamiento
* modelo de clasificación robusto

El resultado es un sistema capaz de **identificar clientes con riesgo de abandono**, permitiendo a la empresa implementar estrategias de retención basadas en datos.

---

# 👨‍💻 Autor

**Jharle Compres**

Ingeniero en Sistemas
Especializado en ciencia de datos y desarrollo de soluciones tecnológicas.
