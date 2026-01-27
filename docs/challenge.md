## Documentacion de lo realizado en el proyecto

primero se realiza la creación de un ambiente virtual para trabajar con las dependencias necesarias, se utiliza uv para la creación de esto 

uv vevn mlops-env

source mlops-env/bin/activate

ya que se teiene activado el ambiente virtual se genera un archivo pyproject.toml para tener las dependencias estrucutradas en un solo archivo.

se instala con el comando: uv pip install -e ".[dev]"


## MODEL

Analizando los modelos que se tienen dentro de exploration.ipynb 

Primer modelo entrenado con XGBoost tiene un problema y es que está detectando todo como clase 0 

INSERTAR IMAGEN

Logra el 81% de accuracy porque la clase 0 es del 81% por lo tanto no detecta nada de clase 1


Primer modelo con regresión logistica

Ligeramente es mejor porque si detecta al manos unos casos de clase 1, pero sigue siendo muy conservador el numero de predicciones.


Ahroa para los modelos que usan el top 10 de features importance y están balanceados se obtiene lo siguiente:

Regresión Logística:

Matriz: [[9487, 8807], [1314, 2900]]
F1 clase 1: 0.36
Recall clase 1: 0.69

XGBoost:

Matriz: [[9556, 8738], [1313, 2901]]
F1 clase 1: 0.37
Recall clase 1: 0.69

practiacamente son lo mismos y desbalanceados no mejora nada así que tomando esto en consideración usaría el XGBoost ya que tiene ventajas como poder ajustar más hiperparámetros  a la hora de querer mejorarlo y en terminos generales es más robusto.


## Creación de clases para el archivo model.py

Se generó un archivo .py del notebook a partir de pytext esto como practica para eficientar la escritura de código. 
A partir de ahi se usaron las clases que ya venían en la plantilla: preprocess, fit y predict con lo que se tenía en el notebook. Además se generaron
funciones de ayuda para el preprocesado como la función _generate_target para generar la variable objetivo; la función _get_min_diff para calcular la diferencia en minutos y además se agregó la función predict_proba para obetner el valor de la probabilidad del restrazo, esto para tener una mejor lectura del código y fuera más estrcuturado. 


## creación de las clases para el archivo api.py
# 📄 API de Predicción de Retrasos de Vuelos

## Descripción de la implementación

Este proyecto implementa una **API RESTful** utilizando **FastAPI** para exponer un modelo de Machine Learning que predice si un vuelo tendrá retraso. La API funciona como la capa de *serving* del modelo entrenado, permitiendo que aplicaciones externas envíen datos de vuelos y obtengan predicciones en tiempo real.

La solución sigue una arquitectura típica de **MLOps**, separando claramente la validación de datos, la lógica del modelo y la exposición vía HTTP.

---

## 🧠 Arquitectura general

La API se organiza en tres capas principales:

1. **Capa de entrada y validación**
   - Implementada con FastAPI y Pydantic.
   - Valida que los datos enviados por el cliente cumplan las reglas del modelo.

2. **Capa de lógica de negocio**
   - Implementada en la clase `DelayModel`.
   - Contiene el pipeline de preprocesamiento y el modelo entrenado.

3. **Capa de exposición**
   - Implementada como endpoints REST (`/health` y `/predict`).

---

## 🚀 Inicialización del modelo

```python
app = fastapi.FastAPI()
model = DelayModel()
