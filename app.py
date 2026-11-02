# Tarea #3 - Dashboards en Python
# Nombre: [JUAN MORALES]
# Grupo 1IL-133

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from metrics import threshold_metrics

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
        title=f"Distribución por diagnóstico<br>{feature}"
    )
    fig.update_layout(title_font_size=14, margin=dict(l=60, r=24, t=80, b=100),
                      legend=dict(orientation="h", y=-0.22, x=0))
    return fig

def build_boxplot(feature):
    fig = px.box(
        df, x="diagnosis", y=feature, color="diagnosis",
        title=f"Boxplot por diagnóstico<br>{feature}"
    )
    fig.update_layout(title_font_size=14, margin=dict(l=60, r=24, t=80, b=100),
                      legend=dict(orientation="h", y=-0.22, x=0))
    return fig

def build_heatmap():
    corr = df[feature_cols].corr()
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", origin="lower",
        title="Mapa de calor de correlación entre variables"
    )
    fig.update_layout(height=640, title_font_size=14,
                      margin=dict(l=170, r=70, t=70, b=170))
    return fig

def build_confusion_matrix(threshold):
    cm, _, _ = threshold_metrics(y_test, y_proba, threshold)
    fig = px.imshow(
        cm, text_auto=True,
        x=["Pred:<br>Maligno", "Pred:<br>Benigno"],
        y=["Real:<br>Maligno", "Real:<br>Benigno"],
        color_continuous_scale="Blues",
        title=f"Matriz de confusión<br>Umbral = {threshold:.2f}"
    )
    fig.update_layout(title_font_size=14, margin=dict(l=70, r=35, t=80, b=80),
                      coloraxis_colorbar=dict(thickness=12))
    fig.update_xaxes(tickangle=0)
    return fig

def build_roc_curve(threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              name=f"ROC (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash"), name="Azar"))

    # El punto usa las mismas predicciones que la matriz de confusión.
    _, current_fpr, current_tpr = threshold_metrics(y_test, y_proba, threshold)
    fig.add_trace(go.Scatter(
        x=[current_fpr], y=[current_tpr], mode="markers",
        marker=dict(size=12, color="red"), name="Umbral actual"
    ))
    fig.update_layout(title="Curva ROC · Regresión logística", title_font_size=14,
                       margin=dict(l=60, r=24, t=70, b=120),
                       legend=dict(orientation="h", y=-0.25, x=0),
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
        html.Label("Selecciona una característica (feature):", htmlFor="feature-dropdown"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": f, "value": f} for f in feature_cols],
            value="mean radius",
            clearable=False
        ),
    ], style={"width": "min(100%, 760px)", "margin": "auto", "padding": "10px"}),

    html.Div([
        dcc.Graph(id="histogram-graph", responsive=True),
        dcc.Graph(id="boxplot-graph", responsive=True),
    ], className="chart-grid"),

    html.P("Correlaciones: en pantallas pequeñas puedes desplazar la matriz horizontalmente."),
    html.Div(dcc.Graph(id="heatmap-graph", figure=build_heatmap(), responsive=True),
             className="correlation-scroll", tabIndex=0,
             role="region", **{"aria-label": "Matriz de correlaciones desplazable"}),

    html.Hr(),
    html.H2("Modelo: Regresión Logística", style={"textAlign": "center"}),

    html.P("Clase positiva: Benigno (1). Se predice Benigno cuando su probabilidad es mayor o igual al umbral."),
    html.P("Falso positivo: maligno clasificado como benigno. Falso negativo: benigno clasificado como maligno."),
    html.P("Ejercicio educativo con datos públicos; no validado para decisiones clínicas."),
    html.Div([
        html.Label("Umbral de decisión (threshold):"),
        dcc.Slider(
            id="threshold-slider",
            min=0.1, max=0.9, step=0.05, value=0.5,
            allow_direct_input=False,
            marks={i / 10: str(i / 10) for i in range(1, 10)}
        ),
    ], style={"width": "min(100%, 760px)", "margin": "auto", "padding": "10px"}),

    html.Div([
        dcc.Graph(id="confusion-graph", responsive=True),
        dcc.Graph(id="roc-graph", responsive=True),
    ], className="chart-grid"),
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
    app.run(debug=False)
