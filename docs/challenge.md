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