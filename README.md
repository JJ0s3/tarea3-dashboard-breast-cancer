# Breast Cancer Wisconsin · Dashboard académico

Aplicación interactiva para explorar datos y entender cómo un umbral de clasificación cambia las predicciones de una regresión logística. Desarrollada como Tarea #3 de Análisis de Datos y Toma de Decisiones en Computación, por Juan Morales.

**Estado:** prototipo académico. Ejercicio educativo, sin validación para uso clínico.

## Qué puedes explorar

- Distribuciones y boxplots de 30 características, separados por diagnóstico.
- Correlaciones entre variables.
- Matriz de confusión y curva ROC con un umbral ajustable.
- Presentación de dos columnas en escritorio y una en móvil; la matriz de correlaciones tiene desplazamiento horizontal propio cuando se necesita.

## Ejecución local

Entorno comprobado: Python 3.12.14 en Linux. Las versiones usadas están fijadas en `requirements.txt`; Windows y otras versiones de Python requieren comprobación adicional.

```bash
python -m venv .venv
```

Activar en PowerShell: `.\.venv\Scripts\Activate.ps1`. En Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python app.py
```

Abrir http://127.0.0.1:8050. El arranque tiene depuración desactivada y está destinado a uso local. No se ha preparado un despliegue público.

## Cómo interpretar el modelo

El dataset integrado de scikit-learn contiene 569 observaciones y 30 características. La variable objetivo usa **0 = Maligno** y **1 = Benigno**. La probabilidad mostrada corresponde a Benigno; se predice esa clase cuando `probabilidad >= umbral`.

| Métrica | Interpretación en esta aplicación |
|---|---|
| Verdadero positivo | Benigno clasificado como benigno |
| Falso positivo | Maligno clasificado como benigno |
| Falso negativo | Benigno clasificado como maligno |
| FPR | FP / (FP + TN) |
| TPR | TP / (TP + FN) |

El marcador ROC se calcula directamente desde las predicciones del umbral seleccionado. Buscar el umbral más cercano entre los puntos de la curva puede producir un marcador distinto, pues la representación ROC puede omitir puntos intermedios.

## Decisiones y límites

Se usa una partición estratificada con 25 % de prueba (143 observaciones), `random_state=42`, escalado ajustado exclusivamente con entrenamiento y regresión logística con `max_iter=5000`. No se realiza validación cruzada, calibración ni evaluación externa. Ajustar el slider sobre el conjunto de prueba es una demostración; no es un procedimiento para seleccionar un umbral clínico.

El AUC se calcula al ejecutar la aplicación y depende del entorno y de esta partición. No se presenta como desempeño generalizable. Dash utiliza Flask como servidor; este proyecto no implementa una API Flask independiente.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Pruebas de tasas conocidas, igualdad en el umbral, entradas inválidas, separación de datos, figuras, rutas HTTP y callback Dash. La comparación ROC/matriz recorre los 17 valores del slider. El workflow incluido ejecutará la suite en GitHub Actions tras su publicación; su resultado remoto aún no se ha comprobado.

## Datos y créditos

Dataset: [Breast Cancer Wisconsin (Diagnostic), UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), cargado mediante [`load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). No se publican datos de pacientes propios. No se añade una licencia de código sin que el titular la decida.

Repositorio: https://github.com/JJ0s3/tarea3-dashboard-breast-cancer
