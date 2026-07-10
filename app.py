# Tarea #3 - Dashboards en Python
# Nombre: [JUAN MORALES]
# Cédula: [8-997-508]
# Grupo 1IL-133

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ------------------------------------------------------------------
# 1. Carga y preparación de datos
# ------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "Maligno", 1: "Benigno"})

feature_cols = list(data.feature_names)

# ------------------------------------------------------------------
# 2. Entrenamiento del modelo (Regresión Logística)
# ------------------------------------------------------------------
X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Probabilidades de la clase "Benigno" (1) para el set de prueba
y_proba = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ------------------------------------------------------------------
# 3. Funciones para construir cada gráfica
# ------------------------------------------------------------------
def build_histogram(feature):
    fig = px.histogram(
        df, x=feature, color="diagnosis", barmode="overlay",
        nbins=30, opacity=0.7,
        title=f"Distribución de {feature} por diagnóstico"
    )
    return fig

def build_boxplot(feature):
    fig = px.box(
        df, x="diagnosis", y=feature, color="diagnosis",
        title=f"Boxplot de {feature} por diagnóstico"
    )
    return fig

def build_heatmap():
    corr = df[feature_cols].corr()
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", origin="lower",
        title="Mapa de calor de correlación entre variables"
    )
    return fig

def build_confusion_matrix(threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, text_auto=True,
        x=["Pred: Maligno", "Pred: Benigno"],
        y=["Real: Maligno", "Real: Benigno"],
        color_continuous_scale="Blues",
        title=f"Matriz de confusión (umbral = {threshold:.2f})"
    )
    return fig

def build_roc_curve(threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              name=f"ROC (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash"), name="Azar"))

    # Punto en la curva correspondiente al umbral seleccionado
    idx = (np.abs(roc_thresholds - threshold)).argmin()
    fig.add_trace(go.Scatter(
        x=[fpr[idx]], y=[tpr[idx]], mode="markers",
        marker=dict(size=12, color="red"), name="Umbral actual"
    ))
    fig.update_layout(title="Curva ROC - Regresión Logística",
                       xaxis_title="Falsos Positivos (FPR)",
                       yaxis_title="Verdaderos Positivos (TPR)")
    return fig

# ------------------------------------------------------------------
# 4. Aplicación Dash / Flask
# ------------------------------------------------------------------
app = Dash(__name__)
server = app.server  # necesario si se despliega en Heroku/PythonAnywhere

app.layout = html.Div([
    html.H1("Dashboard - Breast Cancer Wisconsin", style={"textAlign": "center"}),
    html.P("Tarea #3 - Análisis de Datos y Toma de Decisiones en Computación",
           style={"textAlign": "center"}),

    html.Div([
        html.Label("Selecciona una característica (feature):"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": f, "value": f} for f in feature_cols],
            value="mean radius",
            clearable=False
        ),
    ], style={"width": "50%", "margin": "auto", "padding": "10px"}),

    html.Div([
        dcc.Graph(id="histogram-graph"),
        dcc.Graph(id="boxplot-graph"),
    ], style={"display": "flex"}),

    dcc.Graph(id="heatmap-graph", figure=build_heatmap()),

    html.Hr(),
    html.H2("Modelo: Regresión Logística", style={"textAlign": "center"}),

    html.Div([
        html.Label("Umbral de decisión (threshold):"),
        dcc.Slider(
            id="threshold-slider",
            min=0.1, max=0.9, step=0.05, value=0.5,
            marks={i / 10: str(i / 10) for i in range(1, 10)}
        ),
    ], style={"width": "60%", "margin": "auto", "padding": "10px"}),

    html.Div([
        dcc.Graph(id="confusion-graph"),
        dcc.Graph(id="roc-graph"),
    ], style={"display": "flex"}),
])

# ------------------------------------------------------------------
# 5. Callbacks
# ------------------------------------------------------------------
@app.callback(
    Output("histogram-graph", "figure"),
    Output("boxplot-graph", "figure"),
    Input("feature-dropdown", "value")
)
def update_feature_graphs(feature):
    return build_histogram(feature), build_boxplot(feature)

@app.callback(
    Output("confusion-graph", "figure"),
    Output("roc-graph", "figure"),
    Input("threshold-slider", "value")
)
def update_model_graphs(threshold):
    return build_confusion_matrix(threshold), build_roc_curve(threshold)

# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)