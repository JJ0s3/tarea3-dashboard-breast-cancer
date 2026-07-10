# Tarea #3 - Dashboard con Python (Flask + Dash + Plotly)

## Descripción
Dashboard interactivo para el análisis del dataset Breast Cancer Wisconsin
(Diagnostic), incluyendo EDA y un modelo de Regresión Logística para
clasificar tumores como malignos o benignos.

## Dataset
Breast Cancer Wisconsin (Diagnostic), cargado directamente desde
`sklearn.datasets.load_breast_cancer`.

## Requisitos
- Python 3.10+
- Ver `requirements.txt`

## Instalación y ejecución
1. Crear entorno virtual: `python -m venv venv`
2. Activar entorno:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Instalar dependencias: `pip install -r requirements.txt`
4. Ejecutar: `python app.py`
5. Abrir en el navegador: http://127.0.0.1:8050

## Contenido del dashboard
- Histograma y boxplot de la característica seleccionada, por diagnóstico
- Mapa de calor de correlación entre variables
- Matriz de confusión y curva ROC del modelo de Regresión Logística,
  ajustables según el umbral de decisión (threshold)

## Repositorio
https://github.com/JJ0s3/tarea3-dashboard-breast-cancer