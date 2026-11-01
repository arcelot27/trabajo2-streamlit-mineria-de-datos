"""DataClara: laboratorio de limpieza y análisis con Streamlit."""

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import Options, artist_summary, clean, is_tours, load_csv, normalize, quality
from data_tools import (
    SUPPORTED,
    build_excel,
    build_html_report,
    diagnostics,
    excel_sheets,
    infer_date_settings,
    load_dataset,
    local_datasets,
    profile,
    quality_indicators,
    suggested_types,
)
from studio_ui import advanced_options, general_filters, render_general_analysis

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="DataClara | Análisis de datos", page_icon=":material/analytics:", layout="wide"
)
st.markdown(
    """<style>
.block-container {padding-top:2.5rem; padding-bottom:3rem; max-width:1500px;}
[data-testid="stMetric"] {background:color-mix(in srgb, currentColor 4%, transparent);
border:1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius:16px; padding:20px;}
[data-testid="stMetricLabel"] {opacity:.8;}
[data-testid="stMetricValue"] {font-weight:650; letter-spacing:-1px;}
[data-testid="stTabs"] [role="tablist"] {gap:1rem;}
[data-testid="stTabs"] [role="tab"] {min-height:48px;}
h1 {letter-spacing:-1.5px;} h2 {letter-spacing:-.5px;}
.eyebrow {font-size:.8rem;font-weight:700;letter-spacing:2px;opacity:.8;}
@media (max-width:640px) {.block-container {padding:1rem;} h1 {font-size:2rem;}}
</style>""",
    unsafe_allow_html=True,
)
px.defaults.color_discrete_sequence = None
px.defaults.template = "streamlit"


def chart(fig, name):
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        separators=",.",
    )
    st.plotly_chart(fig, width="stretch", key=name)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def histogram(frame, column, bins=20):
    """Cuenta todos los valores en el servidor; envía solo los intervalos."""
    values = frame[column].dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    fig = go.Figure(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2,
            y=counts,
            width=np.diff(edges),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate="Intervalo: %{customdata[0]:,.3f} a %{customdata[1]:,.3f}<br>Registros: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Registros")
    return fig


def usd(value):
    return "Sin datos" if pd.isna(value) else f"US$ {value:,.0f}"


st.sidebar.title("DataClara")
st.sidebar.caption("LABORATORIO DE DATOS")
st.sidebar.divider()
source = st.sidebar.radio(
    "Fuente de datos",
    ["Dataset de la carpeta", "Subir un archivo", "Demostración de ventas"],
    index=1,
)
uploaded = None
if source == "Subir un archivo":
    uploaded = st.sidebar.file_uploader(
        "Selecciona tu archivo",
        type=sorted(s.lstrip(".") for s in SUPPORTED),
        help="Hasta 50 MB. CSV, TSV, Excel, JSON, JSONL o Parquet. Encabezados únicos.",
    )
    if uploaded is None:
        st.title("Tus datos, más claros")
        st.info("Carga un archivo en el panel lateral para comenzar.")
        st.stop()
    payload, filename = uploaded.getvalue(), uploaded.name
    st.sidebar.success(f"Archivo recibido: {filename}")
elif source == "Demostración de ventas":
    filename = "ventas_demo.csv"
    payload = (ROOT / "sample_data" / filename).read_bytes()
else:
    files = local_datasets(ROOT)
    if not files:
        st.error("No hay datasets en la carpeta software. Usa la opción Subir un archivo.")
        st.stop()
    selected_file = st.sidebar.selectbox(
        "Archivo local", files, format_func=lambda p: str(p.relative_to(ROOT))
    )
    filename, payload = selected_file.name, selected_file.read_bytes()
with st.sidebar.expander("Opciones de lectura"):
    separator = st.selectbox(
        "Separador",
        ["Automático", ";", ",", "\t", "|"],
        format_func=lambda x: "Tabulación" if x == "\t" else x,
    )
    encoding = st.selectbox("Codificación", ["Automático", "utf-8-sig", "cp1252", "latin-1"])
try:
    sheet = 0
    if Path(filename).suffix.lower() in {".xlsx", ".xls"}:
        sheets = excel_sheets(payload)
        sheet = st.sidebar.selectbox("Hoja del libro", sheets)
    signature = hashlib.sha256(
        payload + json.dumps([filename, separator, encoding, sheet]).encode()
    ).hexdigest()[:16]
    if st.session_state.get("loaded_signature") != signature:
        with st.spinner("Leyendo el archivo y sus columnas…", show_time=True):
            if Path(filename).suffix.lower() == ".csv":
                st.session_state["loaded_dataset"] = load_csv(payload, separator, encoding)
            else:
                st.session_state["loaded_dataset"] = load_dataset(
                    payload, filename, separator, encoding, sheet
                )
        for cached_key in [
            "cleaning_result",
            "cleaning_key",
            "normalization_result",
            "normalization_key",
            "type_suggestions",
            "type_suggestion_key",
            "date_suggestions",
            "date_suggestion_key",
            "studio_exports",
            "zip_export",
        ]:
            st.session_state.pop(cached_key, None)
        st.session_state["loaded_signature"] = signature
    original, read_info = st.session_state["loaded_dataset"]
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

loading = st.empty()
loading.info(
    f"Archivo leído: {filename} · {len(original):,} filas · {len(original.columns)} columnas. Preparando el análisis…"
)
with st.expander("Vista previa del archivo cargado"):
    st.dataframe(original.head(100), hide_index=True, width="stretch")
tours_input = is_tours(original)
st.sidebar.caption(f"{len(original):,} registros · {len(original.columns)} columnas")
with st.sidebar.expander("Reglas de limpieza", expanded=False):
    trim = st.checkbox("Normalizar espacios", True)
    dedup = st.checkbox("Eliminar duplicados exactos", True)
    policies = (
        ["Conservar", "Eliminar filas incompletas"]
        if tours_input
        else [
            "Conservar",
            "Mediana / Sin dato",
            "Media / Moda",
            "Cero / Sin dato",
            "Moda",
            "Valor anterior",
            "Eliminar filas incompletas",
        ]
    )
    missing = st.selectbox("Valores faltantes", policies, key=f"missing_{tours_input}")
    if tours_input:
        st.caption(
            "Esquema de giras detectado. Se convierten monedas, posiciones y años. Los datos desconocidos se conservan vacíos por defecto."
        )
        numeric, decimal = [], "."
    else:
        decimal = st.selectbox(
            "Separador decimal",
            [".", ","],
            help="El otro símbolo se interpreta como separador de miles.",
        )
        suggestion_key = (signature, decimal)
        if st.session_state.get("type_suggestion_key") != suggestion_key:
            with st.spinner("Identificando columnas numéricas…", show_time=True):
                st.session_state["type_suggestions"] = suggested_types(original, decimal)
            st.session_state["type_suggestion_key"] = suggestion_key
        suggestions = st.session_state["type_suggestions"]
        recommended = suggestions.loc[suggestions["sugerencia"] == "Número", "columna"].tolist()
        numeric = st.multiselect(
            "Columnas que deben ser numéricas",
            original.columns.tolist(),
            default=recommended,
            key=f"numeric_{signature}",
        )
        st.caption(
            "Sugerencias numéricas preseleccionadas; puedes cambiarlas. Los posibles identificadores se conservan como texto. Los porcentajes se expresan en puntos porcentuales (45% → 45)."
        )
date_defaults = ([], "Día/mes/año")
if not tours_input:
    if st.session_state.get("date_suggestion_key") != suggestion_key:
        st.session_state["date_suggestions"] = infer_date_settings(original, suggestions)
        st.session_state["date_suggestion_key"] = suggestion_key
    date_defaults = st.session_state["date_suggestions"]
advanced = advanced_options(original, tours_input, signature, numeric, date_defaults)
options = Options(dedup, trim, missing, numeric, decimal, **advanced)
try:
    cleaning_key = (signature, json.dumps(asdict(options), sort_keys=True, ensure_ascii=False))
    if st.session_state.get("cleaning_key") != cleaning_key:
        with st.spinner(
            f"Limpiando {len(original):,} filas y registrando los cambios…", show_time=True
        ):
            started = perf_counter()
            st.session_state["cleaning_result"] = clean(original, options)
            st.session_state["cleaning_seconds"] = perf_counter() - started
        st.session_state["cleaning_key"] = cleaning_key
    cleaned, audit, tours = st.session_state["cleaning_result"]
except ValueError as exc:
    st.error(str(exc))
    st.stop()

loading.empty()
st.markdown('<div class="eyebrow">DE LOS DATOS A LAS DECISIONES</div>', unsafe_allow_html=True)
st.title("Tus datos, más claros")
st.write("Limpia, explora y encuentra las señales que importan.")
st.caption(f"Fuente: {filename} · {read_info} · El archivo original se conserva.")
st.caption(
    f"Limpieza: {st.session_state.get('cleaning_seconds', 0):.2f} s · {len(cleaned):,} filas conservadas. Cambiar de sección o filtrar reutiliza los datos procesados."
)
dashboard, cleaning, normalization_tab, general_tab, prediction_tab, explorer, data_tab, export = (
    st.tabs(
        [
            "Resumen ejecutivo",
            "Calidad y limpieza",
            "Normalización",
            "Análisis general",
            "Predicción y decisiones",
            "Explorador",
            "Datos",
            "Descargas",
        ],
        key="active_section",
        on_change="rerun",
    )
)

filtered = cleaned.copy()
metric = "recaudacion_usd"
chosen = []
if tours and not cleaned.empty:
    st.sidebar.divider()
    st.sidebar.subheader("Filtros del análisis")
    artists = sorted(cleaned["artista"].dropna().unique().tolist())
    chosen = st.sidebar.multiselect(
        "Artistas", artists, default=artists, key=f"artists_{signature}"
    )
    filtered = filtered.loc[filtered["artista"].isin(chosen)]
    years = cleaned["anio_inicio"].dropna()
    if not years.empty:
        low = int(years.min())
        high = int(cleaned["anio_fin"].max())
        if low < high:
            bounds = st.sidebar.slider(
                "Periodo de las giras", low, high, (low, high), key=f"years_{signature}"
            )
            include_unknown = st.sidebar.checkbox("Incluir giras sin años válidos", True)
            mask = (filtered["anio_inicio"] <= bounds[1]) & (filtered["anio_fin"] >= bounds[0])
            if include_unknown:
                mask = mask | filtered["anio_inicio"].isna()
            filtered = filtered.loc[mask.fillna(False)]
            st.sidebar.caption(
                "Incluye giras cuyo periodo coincide total o parcialmente con el intervalo."
            )
    if "recaudacion_ajustada_usd" in cleaned:
        mode = st.sidebar.radio("Base monetaria", ["Recaudación nominal", "Ajustada (USD de 2022)"])
        if mode.startswith("Ajustada"):
            metric = "recaudacion_ajustada_usd"
    st.sidebar.caption(
        "Los filtros afectan al resumen, explorador y exportación de la vista. La calidad corresponde a todos los datos limpios."
    )

filtered, general_filter_spec = general_filters(filtered, signature)

# Solo preparar escalado al abrir Normalización o Descargas. Las preferencias
# se guardan fuera del estado de los widgets para conservarlas entre secciones.
numeric_candidates = cleaned.select_dtypes("number").columns.tolist()
default_columns = [
    c
    for c in numeric_candidates
    if c
    not in {"posicion", "mejor_posicion", "mejor_posicion_historica", "anio_inicio", "anio_fin"}
]
preferences = st.session_state.get("normalization_preferences", {}).get(signature, {})
method = preferences.get("method", "Min–Max (0 a 1)")
normalization_columns = [
    c for c in preferences.get("columns", default_columns) if c in numeric_candidates
]
normalized, norm_parameters = None, None
business_spec = st.session_state.get("business_preferences", {}).get(signature, {})
findings, report_figures = [], []

if dashboard.open:
    with dashboard:
        if filtered.empty:
            st.info(
                "No hay registros con esta configuración. Amplía los filtros o conserva los valores faltantes."
            )
        elif tours:
            st.subheader("Panorama de las giras")
            st.caption(
                f"{len(filtered)} de {len(cleaned)} giras · {'USD de 2022, según el archivo' if metric != 'recaudacion_usd' else 'USD nominales'}"
            )
            total = filtered[metric].sum(min_count=1)
            pairs = filtered.dropna(subset=[metric, "conciertos"])
            per_show = (
                pairs[metric].sum() / pairs["conciertos"].sum()
                if pairs["conciertos"].sum() > 0
                else float("nan")
            )
            a, b, c, d = st.columns(4)
            compact_total = (
                "Sin datos"
                if pd.isna(total)
                else f"US$ {total / 1_000_000:,.0f} M".replace(",", ".")
            )
            a.metric(
                "Recaudación de la muestra",
                compact_total,
                help=f"Total exacto: {usd(total)}. M = millones de dólares; la tarjeta redondea al millón.",
            )
            b.metric("Conciertos registrados", f"{filtered['conciertos'].sum():,.0f}")
            c.metric(
                "Ingreso por concierto",
                usd(per_show),
                help="Suma de recaudación / suma de conciertos, usando solo pares completos. No es utilidad.",
            )
            d.metric("Artistas", filtered["artista"].nunique())
            if filtered[[metric, "conciertos"]].isna().any().any():
                st.warning(
                    "Hay importes o conciertos faltantes. Los totales usan datos disponibles y el promedio usa solo pares completos."
                )
            summary = artist_summary(filtered, metric)
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("### ¿Quién concentra la recaudación?")
                if summary["recaudacion"].notna().any():
                    chart(
                        px.bar(
                            summary.sort_values("recaudacion"),
                            x="recaudacion",
                            y="artista",
                            orientation="h",
                            labels={"recaudacion": "Recaudación (USD)", "artista": "Artista"},
                            hover_data=["giras", "conciertos"],
                        ),
                        "artists",
                    )
                else:
                    st.info("No hay recaudación válida para comparar artistas.")
            with right:
                st.markdown("### Escala y rendimiento")
                chart(
                    px.scatter(
                        filtered.dropna(subset=["conciertos", metric]),
                        x="conciertos",
                        y=metric,
                        color="artista",
                        hover_name="gira",
                        labels={
                            metric: "Recaudación (USD)",
                            "conciertos": "Conciertos",
                            "artista": "Artista",
                        },
                    ),
                    "scale",
                )
            st.markdown("### Giras con mayor recaudación")
            top = filtered.nlargest(min(10, len(filtered)), metric).sort_values(metric)
            chart(
                px.bar(
                    top,
                    x=metric,
                    y="gira",
                    color="artista",
                    orientation="h",
                    labels={metric: "Recaudación (USD)", "gira": "Gira", "artista": "Artista"},
                ),
                "ranking",
            )
            st.markdown("### Recaudación por año de inicio")
            annual = (
                filtered.dropna(subset=["anio_inicio"])
                .groupby("anio_inicio", as_index=False)[metric]
                .sum(min_count=1)
            )
            chart(
                px.bar(
                    annual,
                    x="anio_inicio",
                    y=metric,
                    labels={
                        "anio_inicio": "Año de inicio",
                        metric: "Recaudación total de giras (USD)",
                    },
                ),
                "years",
            )
            st.caption(
                "Se atribuye la recaudación completa al año de inicio de cada gira. No representa ingresos anuales ni una serie del mercado."
            )
            st.markdown("### Señales para revisar")
            leaders = summary.dropna(subset=["recaudacion"])
            if not leaders.empty and pd.notna(total) and total > 0:
                leader = leaders.iloc[0]
                st.info(
                    f"{leader['artista']} concentra {leader['recaudacion'] / total:.1%} de la recaudación filtrada. Úsalo para evaluar la concentración de la selección."
                )
            best = summary.dropna(subset=["por_concierto"]).sort_values(
                "por_concierto", ascending=False
            )
            if not best.empty:
                st.success(
                    f"{best.iloc[0]['artista']} presenta el mayor ingreso por concierto: {usd(best.iloc[0]['por_concierto'])}. Compara capacidad de convocatoria y número de fechas antes de planificar eventos."
                )
            st.warning(
                f"La muestra contiene {len(original)} giras en el archivo original; no representa todo el mercado. La recaudación no equivale a rentabilidad: faltan costos, asistencia y precios de entradas."
            )
            st.caption(
                "La columna ajustada se utiliza tal como viene en el CSV; no se recalcula ni se verifica su metodología de inflación."
            )
        else:
            st.subheader("Panorama del dataset")
            a, b, c = st.columns(3)
            a.metric("Registros limpios", len(filtered))
            b.metric("Columnas", len(filtered.columns))
            c.metric("Celdas faltantes", int(filtered.isna().sum().sum()))
            nums = filtered.select_dtypes("number").columns.tolist()
            if nums:
                st.dataframe(filtered[nums].describe().T, width="stretch")
                selected = st.selectbox("Variable para el resumen", nums)
                chart(histogram(filtered, selected), "generic_summary")
            else:
                st.info(
                    "Selecciona las columnas numéricas en Reglas de limpieza para analizar sus distribuciones. En el explorador también puedes contar categorías."
                )

if cleaning.open:
    with cleaning:
        st.subheader("Calidad visible, cambios trazables")
        a, b, c, d = st.columns(4)
        a.metric("Filas originales", len(original))
        b.metric("Filas eliminadas", len(original) - len(cleaned))
        c.metric("Faltantes originales", int(original.isna().sum().sum()))
        d.metric(
            "Faltantes tras limpiar",
            int(cleaned.isna().sum().sum()),
            help="Incluye columnas derivadas y valores inválidos convertidos a vacíos.",
        )
        st.write(
            "Las reglas del panel lateral se aplican a una copia. Las cifras ambiguas quedan vacías. Por defecto se conservan los atípicos; puedes configurar su tratamiento en Limpieza avanzada."
        )
        indicators_before, indicators_after = (
            quality_indicators(original),
            quality_indicators(cleaned),
        )
        st.metric(
            "Indicador de completitud y unicidad",
            f"{indicators_after['puntaje']:.1f} / 100",
            delta=f"{indicators_after['puntaje'] - indicators_before['puntaje']:+.1f} puntos",
        )
        st.caption(
            "Indicador descriptivo = 70% completitud + 30% unicidad de filas. No mide exactitud, representatividad ni validez del negocio. Agregar columnas derivadas puede cambiarlo."
        )
        with st.expander("Diagnóstico y sugerencias de tipos"):
            st.dataframe(diagnostics(original), hide_index=True, width="stretch")
            st.dataframe(
                suggested_types(original, decimal) if tours_input else suggestions,
                hide_index=True,
                width="stretch",
            )
        with st.expander("Perfil estadístico completo"):
            st.dataframe(profile(cleaned), hide_index=True, width="stretch")
        before, after = st.columns(2)
        with before:
            st.markdown("### Antes")
            st.dataframe(quality(original), hide_index=True, width="stretch")
        with after:
            st.markdown("### Después")
            st.dataframe(quality(cleaned), hide_index=True, width="stretch")
        st.caption(
            "La tabla de calidad cuenta atípicos con 1,5 × IQR cuando IQR > 0. Limpieza avanzada permite elegir otro método y factor para su tratamiento. Son señales estadísticas, no pruebas de error."
        )
        if tours:
            st.info(
                "El ingreso por concierto se calcula con recaudación / conciertos y se guarda separado del promedio reportado. Por ejemplo, el formato '$ 615,39' no se interpreta como una cifra estadounidense válida."
            )
            st.caption(
                "Las referencias bibliográficas permanecen en su columna. Las posiciones faltantes no se inventan. La fila del registro de cambios corresponde a la línea del CSV original (encabezado = 1; sin campos multilínea)."
            )
        st.markdown("### Registro de transformaciones")
        if audit.empty:
            st.success("No se necesitaron transformaciones con las reglas actuales.")
        else:
            st.dataframe(audit.head(1000), hide_index=True, width="stretch")
            if options.audit_mode == "Resumido" or (
                options.audit_mode == "Automático" and len(original) > 50_000
            ):
                st.caption(
                    "Auditoría resumida por columna y operación. Cantidad indica las celdas o filas afectadas; antes/después son un ejemplo. No contiene una fila por cada celda cambiada."
                )
            if len(audit) > 1000:
                st.caption(
                    f"Se muestran los primeros 1.000 de {len(audit):,} cambios. La descarga conserva el registro completo."
                )

if normalization_tab.open:
    with normalization_tab:
        st.subheader("Variables en una escala comparable")
        st.write(
            "Normaliza las columnas numéricas después de limpiar el dataset. Las columnas de texto se conservan y los datos faltantes permanecen vacíos."
        )
        numeric_candidates = cleaned.select_dtypes("number").columns.tolist()
        default_columns = [
            c
            for c in numeric_candidates
            if c
            not in {
                "posicion",
                "mejor_posicion",
                "mejor_posicion_historica",
                "anio_inicio",
                "anio_fin",
            }
        ]
        method = st.radio(
            "Método de normalización",
            ["Min–Max (0 a 1)", "Z-score"],
            index=0 if method.startswith("Min") else 1,
            horizontal=True,
        )
        normalization_columns = st.multiselect(
            "Columnas a normalizar",
            numeric_candidates,
            default=normalization_columns,
            key=f"normalize_{signature}",
        )
        st.session_state.setdefault("normalization_preferences", {})[signature] = {
            "method": method,
            "columns": normalization_columns,
        }
        normalization_key = (cleaning_key, method, tuple(normalization_columns))
        if st.session_state.get("normalization_key") != normalization_key:
            st.session_state["normalization_result"] = normalize(
                cleaned, normalization_columns, method
            )
            st.session_state["normalization_key"] = normalization_key
        normalized, norm_parameters = st.session_state["normalization_result"]
        st.caption(
            "Se calculan los parámetros con todos los datos limpios antes de los filtros. Los indicadores del resumen mantienen sus unidades originales. Rankings y años no se seleccionan por defecto."
        )
        if normalization_columns:
            st.code(
                "x_normalizado = (x − mínimo) / (máximo − mínimo)"
                if method.startswith("Min")
                else "z = (x − media) / desviación poblacional (ddof=0)",
                language=None,
            )
            a, b, c = st.columns(3)
            a.metric("Variables normalizadas", len(normalization_columns))
            b.metric("Registros", len(normalized))
            c.metric("Método", "Min–Max" if method.startswith("Min") else "Z-score")
            variable = st.selectbox(
                "Variable para comparar", normalization_columns, key=f"norm_compare_{signature}"
            )
            left, right = st.columns(2)
            with left:
                st.markdown("### Escala original")
                chart(histogram(cleaned, variable, 15), "norm_before")
            with right:
                st.markdown("### Escala normalizada")
                chart(histogram(normalized, variable, 15), "norm_after")
            st.markdown("### Dataset normalizado completo")
            st.dataframe(normalized.head(1000), hide_index=True, width="stretch")
            st.caption(
                f"Vista previa: hasta 1.000 de {len(normalized):,} filas. La descarga incluye el dataset completo."
            )
            with st.expander("Parámetros y tratamiento de casos especiales"):
                st.dataframe(norm_parameters, hide_index=True, width="stretch")
                st.caption(
                    "Las columnas constantes toman el valor 0 en sus celdas válidas. Las columnas completamente vacías siguen vacías. Z-score puede producir valores negativos y no está limitado a [0, 1]. Si entrenas un modelo predictivo, ajusta el escalador solo con su conjunto de entrenamiento."
                )
            st.download_button(
                "Descargar dataset normalizado",
                partial(csv_bytes, normalized),
                "datos_normalizados.csv",
                "text/csv",
                key="normalized_tab_download",
                type="primary",
            )
        else:
            st.info(
                "Selecciona columnas numéricas para normalizar. En otros CSV, primero indica sus tipos en Reglas de limpieza."
            )

if general_tab.open:
    with general_tab:
        findings, report_figures, business_spec = render_general_analysis(
            filtered, chart, signature
        )
        st.session_state.setdefault("business_preferences", {})[signature] = business_spec

if prediction_tab.open:
    with prediction_tab:
        from forecasting import render_forecast

        render_forecast(filtered, chart, signature)

if explorer.open:
    with explorer:
        st.subheader("Hazle preguntas a tus datos")
        st.caption(
            "Trabaja con la selección filtrada. Las gráficas describen asociaciones; no prueban causas."
        )
        nums = filtered.select_dtypes("number").columns.tolist()
        cats = [
            c
            for c in filtered
            if not pd.api.types.is_numeric_dtype(filtered[c])
            and not pd.api.types.is_datetime64_any_dtype(filtered[c])
        ]
        kind = st.selectbox(
            "Tipo de gráfica",
            [
                "Barras por categoría",
                "Distribución",
                "Dispersión",
                "Caja y atípicos",
                "Correlación",
            ],
        )
        if filtered.empty:
            st.info("No hay registros para graficar.")
        elif kind == "Barras por categoría":
            if not cats:
                st.info("No hay columnas categóricas.")
            else:
                category = st.selectbox("Categoría", cats)
                aggregate = st.selectbox(
                    "Medida",
                    ["Conteo de registros", "Suma", "Promedio"]
                    if nums
                    else ["Conteo de registros"],
                )
                top_n = st.slider("Máximo de categorías", 3, 30, 10)
                if aggregate == "Conteo de registros":
                    grouped = (
                        filtered.groupby(category, dropna=False).size().reset_index(name="valor")
                    )
                else:
                    measure = st.selectbox("Variable numérica", nums)
                    group = filtered.groupby(category, dropna=False)[measure]
                    grouped = (
                        group.sum(min_count=1) if aggregate == "Suma" else group.mean()
                    ).reset_index(name="valor")
                grouped[category] = grouped[category].astype("string").fillna("(Sin dato)")
                chart(
                    px.bar(
                        grouped.nlargest(top_n, "valor").sort_values("valor"),
                        x="valor",
                        y=category,
                        orientation="h",
                        labels={"valor": aggregate},
                    ),
                    "custom_bar",
                )
        elif not nums:
            st.info("Configura al menos una columna numérica en Reglas de limpieza.")
        elif kind == "Distribución":
            x = st.selectbox("Variable", nums)
            bins = st.slider("Número de intervalos", 5, 50, 15)
            chart(histogram(filtered, x, bins), "hist")
        elif kind == "Caja y atípicos":
            y = st.selectbox("Variable", nums)
            group = st.selectbox("Agrupar por", ["Sin agrupación"] + cats)
            plot_rows = filtered.sample(5000, random_state=42) if len(filtered) > 5000 else filtered
            chart(
                px.box(
                    plot_rows, y=y, x=None if group == "Sin agrupación" else group, points="all"
                ),
                "box",
            )
            if len(filtered) > 5000:
                st.caption(
                    "La caja y sus puntos usan una muestra reproducible de 5.000 filas para agilizar el navegador. Las tablas y descargas conservan todas las filas."
                )
        elif kind == "Dispersión":
            x = st.selectbox("Eje horizontal", nums)
            y = st.selectbox("Eje vertical", nums, index=min(1, len(nums) - 1))
            color = st.selectbox("Color por", ["Sin agrupación"] + cats)
            plot_rows = filtered.sample(5000, random_state=42) if len(filtered) > 5000 else filtered
            chart(
                px.scatter(plot_rows, x=x, y=y, color=None if color == "Sin agrupación" else color),
                "scatter",
            )
            if len(filtered) > 5000:
                st.caption(
                    "La dispersión muestra una muestra reproducible de 5.000 filas; los cálculos y descargas mantienen el dataset completo."
                )
        else:
            if len(nums) < 2:
                st.info("Se necesitan dos variables numéricas.")
            else:
                corr = filtered[nums].corr(min_periods=3)
                chart(
                    px.imshow(
                        corr,
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        text_auto=".2f",
                        aspect="auto",
                    ),
                    "corr",
                )
                st.caption(
                    "Pearson, con al menos 3 pares válidos. Variables constantes o sin pares suficientes no tienen coeficiente definido. Los rankings y años también pueden aparecer; interpreta su significado antes de comparar."
                )

if data_tab.open:
    with data_tab:
        st.subheader("Inspecciona la información")
        view = st.radio("Vista", ["Datos limpios filtrados", "Original"], horizontal=True)
        table = original if view == "Original" else filtered
        search = st.text_input("Buscar texto en cualquier columna")
        if search:
            mask = (
                table.astype("string")
                .apply(lambda s: s.str.contains(search, case=False, regex=False, na=False))
                .any(axis=1)
            )
            table = table.loc[mask]
        cols = st.multiselect(
            "Columnas visibles",
            table.columns.tolist(),
            default=table.columns.tolist(),
            key=f"visible_{view}_{signature}",
        )
        st.caption(f"{len(table):,} registros en esta vista. La búsqueda solo afecta a esta tabla.")
        page_size = st.selectbox("Filas por página", [100, 500, 1000])
        page_count = max(1, (len(table) + page_size - 1) // page_size)
        page = st.number_input(
            "Página",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"page_{signature}_{len(table)}_{page_size}",
        )
        offset = (page - 1) * page_size
        st.dataframe(
            table[cols].iloc[offset : offset + page_size], hide_index=True, width="stretch"
        )

if export.open:
    with export:
        normalization_key = (cleaning_key, method, tuple(normalization_columns))
        if st.session_state.get("normalization_key") != normalization_key:
            st.session_state["normalization_result"] = normalize(
                cleaned, normalization_columns, method
            )
            st.session_state["normalization_key"] = normalization_key
        normalized, norm_parameters = st.session_state["normalization_result"]
        from data_tools import analysis_snapshot

        findings, report_figures, business_spec = analysis_snapshot(filtered, business_spec)
        st.subheader("Lleva los resultados contigo")
        st.write(
            "Descarga los datos limpios y el registro de cambios para continuar el análisis o presentar tu trabajo."
        )
        report = [
            "# DataClara — Informe de análisis",
            f"Fuente: {filename}",
            f"SHA-256 del original: {hashlib.sha256(payload).hexdigest()}",
            f"Filas originales: {len(original)}",
            f"Filas limpias: {len(cleaned)}",
            f"Filas filtradas: {len(filtered)}",
            f"Faltantes originales: {original.isna().sum().sum()}",
            f"Faltantes limpios: {cleaned.isna().sum().sum()}",
            "## Configuración",
            json.dumps({**asdict(options), "lectura": read_info}, ensure_ascii=False),
            "## Filtros generales",
            json.dumps(general_filter_spec, ensure_ascii=False),
            "## Variables del análisis general",
            json.dumps(business_spec, ensure_ascii=False),
            "## Metodología",
            "Valores numéricos ambiguos se convierten a vacíos. Atípicos según configuración explícita; por defecto se conservan. Las fechas inválidas quedan vacías.",
            "## Normalización",
            f"Método: {method}. Columnas: {', '.join(normalization_columns) or 'Ninguna'}.",
            "Parámetros calculados sobre el dataset limpio completo antes de los filtros. Constantes a cero; faltantes conservados. Z-score usa desviación poblacional (ddof=0). Resumen ejecutivo en unidades originales.",
        ]
        if tours:
            report += [
                f"Base monetaria de la vista: {metric}",
                f"Artistas seleccionados: {', '.join(chosen)}",
                "Periodos de la vista: "
                + ", ".join(filtered["periodo"].dropna().unique().tolist()),
                f"Recaudación filtrada: {usd(filtered[metric].sum(min_count=1))}",
                "Ingreso por concierto = recaudación / conciertos; no equivale a utilidad. Los promedios reportados se conservan por separado.",
                "## Límites",
                "Muestra seleccionada de giras, no mercado completo. Recaudación de giras atribuida al año de inicio; no es ingreso anual. Ajuste a USD de 2022 tomado del archivo, sin recalcular inflación. No hay costos para estimar rentabilidad.",
            ]
            if not filtered.empty:
                report += [
                    "## Comparación por artista",
                    artist_summary(filtered, metric).to_csv(index=False),
                ]
        report += ["## Hallazgos de la vista filtrada", "\n".join("- " + text for text in findings)]
        report_text = "\n\n".join(report)
        a, b, c = st.columns(3)
        a.download_button(
            "Dataset limpio completo", partial(csv_bytes, cleaned), "datos_limpios.csv", "text/csv"
        )
        b.download_button(
            "Vista filtrada", partial(csv_bytes, filtered), "vista_filtrada.csv", "text/csv"
        )
        c.download_button(
            "Registro de cambios", partial(csv_bytes, audit), "registro_limpieza.csv", "text/csv"
        )
        if normalization_columns:
            st.download_button(
                "Dataset normalizado completo",
                partial(csv_bytes, normalized),
                "datos_normalizados.csv",
                "text/csv",
            )
            st.download_button(
                "Parámetros de normalización",
                csv_bytes(norm_parameters),
                "parametros_normalizacion.csv",
                "text/csv",
            )
        st.download_button(
            "Informe de análisis (.md)",
            report_text.encode("utf-8"),
            "informe_analisis.md",
            "text/markdown",
        )
        export_signature = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        if st.button(
            "Generar Excel e informe HTML",
            help="Incluye tablas, auditoría, datos normalizados y gráficas. El HTML funciona sin internet.",
        ):
            try:
                with st.spinner("Preparando los entregables…"):
                    frames = {
                        "Datos limpios": cleaned,
                        "Vista filtrada": filtered,
                        "Perfil": profile(cleaned),
                        "Calidad original": quality(original),
                        "Registro limpieza": audit,
                        "Configuración": pd.DataFrame({"informe": [report_text]}),
                    }
                    if normalization_columns:
                        frames.update(
                            {"Datos normalizados": normalized, "Parámetros": norm_parameters}
                        )
                    excel_data, excel_error = None, None
                    try:
                        excel_data = build_excel(frames)
                    except ValueError as exc:
                        excel_error = str(exc)
                    html_data = build_html_report(
                        filename, report_text, filtered, audit, findings, report_figures
                    ).encode("utf-8")
                    parquet_buffer = io.BytesIO()
                    cleaned.to_parquet(parquet_buffer, index=False)
                    st.session_state["studio_exports"] = {
                        "signature": export_signature,
                        "excel": excel_data,
                        "excel_error": excel_error,
                        "html": html_data,
                        "parquet": parquet_buffer.getvalue(),
                    }
            except (ValueError, TypeError, ImportError) as exc:
                st.error(f"No se pudo generar la exportación: {exc}")
        generated = st.session_state.get("studio_exports", {})
        export_ready = generated.get("signature") == export_signature
        if export_ready:
            if generated["excel"] is not None:
                st.download_button(
                    "Libro Excel (.xlsx)",
                    generated["excel"],
                    "informe_dataclara.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info(
                    generated.get(
                        "excel_error", "Utiliza CSV o Parquet para exportar este dataset."
                    )
                )
            st.download_button(
                "Informe interactivo (.html)",
                generated["html"],
                "informe_dataclara.html",
                "text/html",
            )
            st.download_button(
                "Datos limpios (.parquet)",
                generated["parquet"],
                "datos_limpios.parquet",
                "application/octet-stream",
            )
            st.caption("Para obtener PDF, abre el informe HTML y usa Imprimir → Guardar como PDF.")
        st.download_button(
            "Datos limpios (.json)",
            partial(cleaned.to_json, orient="records", date_format="iso", force_ascii=False),
            "datos_limpios.json",
            "application/json",
        )
        if st.button(
            "Preparar paquete completo (.zip)",
            help="Se genera al solicitarlo para agilizar la carga del dataset.",
        ):
            with st.spinner("Comprimiendo datos e informes…", show_time=True):
                bundle = io.BytesIO()
                with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                    for name, frame in [
                        ("datos_limpios.csv", cleaned),
                        ("vista_filtrada.csv", filtered),
                        ("registro_limpieza.csv", audit),
                        ("calidad_original.csv", quality(original)),
                        ("calidad_final.csv", quality(cleaned)),
                    ]:
                        archive.writestr(name, csv_bytes(frame))
                    archive.writestr("informe_analisis.md", report_text)
                    archive.writestr("diagnostico_original.csv", csv_bytes(diagnostics(original)))
                    archive.writestr("perfil_final.csv", csv_bytes(profile(cleaned)))
                    if export_ready:
                        if generated["excel"] is not None:
                            archive.writestr("informe_dataclara.xlsx", generated["excel"])
                        archive.writestr("informe_dataclara.html", generated["html"])
                        archive.writestr("datos_limpios.parquet", generated["parquet"])
                    if normalization_columns:
                        archive.writestr("datos_normalizados.csv", csv_bytes(normalized))
                        archive.writestr("parametros_normalizacion.csv", csv_bytes(norm_parameters))
                st.session_state["zip_export"] = (export_signature, export_ready, bundle.getvalue())
        saved_zip = st.session_state.get("zip_export")
        if saved_zip and saved_zip[:2] == (export_signature, export_ready):
            st.download_button(
                "Descargar paquete completo (.zip)",
                saved_zip[2],
                "dataclara_resultados.zip",
                "application/zip",
                type="primary",
            )
        st.caption(
            "Los CSV usan UTF-8 con BOM. El paquete refleja las reglas y filtros activos; el dataset limpio completo incluye todas las filas conservadas."
        )
        if not export_ready:
            st.caption(
                "Pulsa Generar Excel e informe HTML para incluir esos formatos y Parquet en el ZIP con la configuración actual."
            )

st.divider()
st.caption("DataClara · Python + Streamlit · Limpieza trazable y análisis exploratorio")
