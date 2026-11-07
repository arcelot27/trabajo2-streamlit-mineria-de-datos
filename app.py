"""Interfaz esencial para limpiar, analizar y proyectar datos del usuario."""

import hashlib
import io
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    clean,
    load_dataset,
    normalize,
    product_sales,
    quality,
    suggest_dates,
    suggest_numeric,
)
from forecasting import (
    FREQUENCIES,
    METHODS,
    aggregate_series,
    forecast,
    forecast_figure,
)

st.set_page_config(
    page_title="DataClara | Minería de datos",
    page_icon=":material/analytics:",
    layout="wide",
)
st.title("DataClara · Minería de datos")
st.caption("Carga tu archivo, revisa la limpieza y analiza sus resultados.")
uploaded = st.sidebar.file_uploader("Dataset CSV o Excel", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Sube un dataset desde el panel lateral para comenzar.")
    st.stop()

payload = uploaded.getvalue()
sheet, separator = 0, "Automático"
try:
    if uploaded.name.lower().endswith(".xlsx"):
        with pd.ExcelFile(io.BytesIO(payload)) as book:
            sheet = st.sidebar.selectbox("Hoja de Excel", book.sheet_names)
    else:
        separator = st.sidebar.selectbox(
            "Separador CSV", ["Automático", ",", ";", "\t", "|"]
        )
    signature = hashlib.sha256(payload + str((sheet, separator)).encode()).hexdigest()
    if (
        st.session_state.get("source") != signature
        or "date_suggestions" not in st.session_state
    ):
        with st.spinner("Leyendo dataset…"):
            original = load_dataset(payload, uploaded.name, separator, sheet)
        for key in [
            "cleaned",
            "audit",
            "options",
            "csv",
            "normalized",
            "normalization_key",
            "product_summary",
            "product_summary_key",
        ]:
            st.session_state.pop(key, None)
        st.session_state.update(
            source=signature,
            original=original,
            numeric_suggestions=suggest_numeric(original),
            date_suggestions=suggest_dates(original),
        )
    original = st.session_state.original
except (ValueError, OSError, ImportError, BadZipFile) as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

st.sidebar.caption(f"{len(original):,} filas · {len(original.columns)} columnas")
formats = {
    "Día/mes/año": "%d/%m/%Y",
    "Mes/día/año": "%m/%d/%Y",
    "Día/mes/año con hora": "%d/%m/%Y %H:%M",
    "Mes/día/año con hora": "%m/%d/%Y %H:%M",
    "Día/mes/año con segundos": "%d/%m/%Y %H:%M:%S",
    "Mes/día/año con segundos": "%m/%d/%Y %H:%M:%S",
    "ISO 8601": "ISO8601",
    "Mixto (día primero)": "mixed_dayfirst",
}
with st.sidebar.form(f"rules_{signature}"):
    st.subheader("Reglas de limpieza")
    decimal = st.selectbox("Separador decimal", [".", ","])
    numeric = st.multiselect(
        "Columnas numéricas",
        original.columns,
        default=st.session_state.numeric_suggestions,
    )
    proposed_dates, proposed_format = st.session_state.date_suggestions
    dates = st.multiselect(
        "Columnas de fecha", original.columns, default=proposed_dates
    )
    date_format = st.selectbox(
        "Formato de fechas",
        list(formats),
        index=list(formats.values()).index(proposed_format),
    )
    if proposed_dates:
        st.caption(
            "Fechas preseleccionadas a partir de una muestra distribuida del archivo. Revisa el formato antes de aplicar cambios."
        )
    st.caption(
        "Confirma el orden de día y mes. Los identificadores deben conservarse como texto."
    )
    deduplicate = st.checkbox("Eliminar duplicados", True)
    lowercase = st.checkbox("Convertir texto a minúsculas")
    missing = st.selectbox(
        "Valores faltantes", ["Conservar", "Mediana / Sin dato", "Eliminar filas"]
    )
    remove_null = st.checkbox("Eliminar columnas con muchos nulos")
    threshold = st.slider("Umbral de nulos (%)", 10, 100, 80)
    outliers = st.selectbox("Valores atípicos", ["Conservar", "Recortar (IQR)"])
    submitted = st.form_submit_button("Aplicar limpieza", type="primary")

options = {
    "numeric": numeric,
    "dates": dates,
    "date_format": formats[date_format],
    "decimal": decimal,
    "deduplicate": deduplicate,
    "lowercase": lowercase,
    "missing": missing,
    "null_threshold": threshold / 100 if remove_null else 1.01,
    "outliers": outliers,
}
if submitted or "cleaned" not in st.session_state:
    if set(numeric) & set(dates):
        st.error("Una columna no puede ser numérica y fecha a la vez.")
        st.stop()
    with st.spinner("Limpiando dataset…"):
        start = perf_counter()
        cleaned, audit = clean(original, options)
        st.session_state.update(
            cleaned=cleaned,
            audit=audit,
            options=options,
            seconds=perf_counter() - start,
        )
        for key in [
            "csv",
            "normalized",
            "normalization_key",
            "product_summary",
            "product_summary_key",
        ]:
            st.session_state.pop(key, None)
cleaned, audit = st.session_state.cleaned, st.session_state.audit
st.caption(
    f"{uploaded.name} · {len(cleaned):,} filas conservadas · Limpieza: {st.session_state.seconds:.2f} s"
)
section = st.sidebar.radio(
    "Sección", ["Limpieza", "Normalización", "Análisis", "Predicción"]
)


def show_chart(figure):
    st.plotly_chart(figure, width="stretch")


if section == "Limpieza":
    a, b, c = st.columns(3)
    a.metric("Filas originales", len(original))
    b.metric("Filas eliminadas", len(original) - len(cleaned))
    c.metric("Faltantes después", int(cleaned.isna().sum().sum()))
    before, after = st.columns(2)
    with before:
        st.subheader("Antes")
        st.dataframe(original.head(100), hide_index=True)
        st.dataframe(quality(original), hide_index=True)
    with after:
        st.subheader("Después")
        st.dataframe(cleaned.head(100), hide_index=True)
        st.dataframe(quality(cleaned), hide_index=True)
    st.caption(
        "Vista previa de hasta 100 filas. El original se conserva; la descarga contiene todas las filas limpias."
    )
    st.subheader("Registro de limpieza")
    st.dataframe(audit, hide_index=True)
    st.caption(
        "El registro resume cantidades por operación. Las conversiones inválidas quedan vacías; los atípicos se conservan salvo que elijas recortarlos."
    )
    if st.button("Preparar CSV limpio"):
        st.session_state.csv = cleaned.to_csv(index=False).encode("utf-8-sig")
    if "csv" in st.session_state:
        st.download_button(
            "Descargar dataset limpio",
            st.session_state.csv,
            "datos_limpios.csv",
            "text/csv",
        )

elif section == "Normalización":
    columns = st.multiselect(
        "Variables numéricas a normalizar", cleaned.select_dtypes("number").columns
    )
    method = st.selectbox("Método", ["Min–Max (0 a 1)", "Z-score"])
    st.caption(
        "Se cambia la escala en una copia: Min–Max lleva los valores a [0, 1]; Z-score usa media y desviación. Los faltantes se conservan."
    )
    if columns:
        key = (tuple(columns), method)
        if st.session_state.get("normalization_key") != key:
            st.session_state.normalized = normalize(cleaned, columns, method)
            st.session_state.normalization_key = key
        normalized, parameters = st.session_state.normalized
        st.dataframe(normalized.head(100), hide_index=True)
        st.dataframe(parameters, hide_index=True)
        if st.button("Preparar CSV normalizado"):
            st.download_button(
                "Descargar dataset normalizado",
                normalized.to_csv(index=False).encode("utf-8-sig"),
                "datos_normalizados.csv",
                "text/csv",
                on_click="ignore",
            )

elif section == "Análisis":
    if cleaned.empty or not len(cleaned.columns):
        st.info("No hay datos con estas reglas. Ajusta la limpieza.")
        st.stop()
    nums = cleaned.select_dtypes("number").columns.tolist()
    categories = cleaned.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    cardinalities = {col: cleaned[col].nunique() for col in categories}
    categories.sort(
        key=lambda col: (not 2 <= cardinalities[col] <= 50, cardinalities[col])
    )
    names = {c.casefold().replace("_", "").replace(" ", ""): c for c in cleaned}
    quantity = next(
        (names[n] for n in ["quantity", "cantidad", "unidades"] if n in names), None
    )
    product = next(
        (
            names[n]
            for n in [
                "stockcode",
                "producto",
                "description",
                "descripcion",
                "descripción",
            ]
            if n in names
        ),
        None,
    )
    kinds = ["Barras por categoría", "Distribución", "Dispersión", "Correlación"]
    kinds.insert(0 if quantity and product else len(kinds), "Productos más vendidos")
    kind = st.selectbox("Gráfica", kinds)
    if kind == "Productos más vendidos" and nums:
        with st.expander(
            "Columnas para el resumen de productos",
            expanded=not bool(quantity and product),
        ):
            product = st.selectbox(
                "Columna de producto o código",
                cleaned.columns,
                index=cleaned.columns.get_loc(product) if product else 0,
            )
            quantity = st.selectbox(
                "Columna de cantidad",
                nums,
                index=nums.index(quantity) if quantity in nums else 0,
            )
            optional = ["Sin seleccionar"] + cleaned.columns.tolist()
            description = (
                names.get("description")
                or names.get("descripcion")
                or names.get("descripción")
            )
            description = st.selectbox(
                "Descripción del producto (opcional)",
                optional,
                index=optional.index(description) if description else 0,
            )
            prices = ["Sin seleccionar"] + nums
            price = (
                names.get("unitprice")
                or names.get("preciounitario")
                or names.get("precio")
            )
            price = st.selectbox(
                "Precio por unidad (opcional)",
                prices,
                index=prices.index(price) if price in nums else 0,
            )
        description = None if description == "Sin seleccionar" else description
        price = None if price == "Sin seleccionar" else price
        summary_key = (product, quantity, price, description)
        if st.session_state.get("product_summary_key") != summary_key:
            st.session_state.product_summary = product_sales(
                cleaned, product, quantity, price, description
            )
            st.session_state.product_summary_key = summary_key
        products, totals = st.session_state.product_summary
        if products.empty:
            st.info(
                "No hay filas con producto y cantidad válidos. Revisa las columnas seleccionadas y la limpieza."
            )
            st.stop()
        st.subheader("Ventas por producto")
        st.caption("Resumen de todo el dataset limpio cargado.")
        a, b, c, d = st.columns(4)
        a.metric("Productos diferentes", len(products))
        b.metric("Unidades netas", f"{totals['unidades_netas']:,.2f}")
        c.metric("Unidades positivas", f"{totals['unidades_vendidas']:,.2f}")
        d.metric(
            "Unidades devueltas o ajustadas", f"{totals['unidades_devueltas']:,.2f}"
        )
        measures = {
            "Unidades netas": "unidades_netas",
            "Unidades positivas (antes de devoluciones)": "unidades_vendidas",
            "Unidades devueltas o ajustadas": "unidades_devueltas",
        }
        if price:
            value = totals["importe_neto"]
            st.metric(
                "Importe neto registrado",
                f"{value:,.2f}" if pd.notna(value) else "Sin datos",
            )
            measures["Importe neto"] = "importe_neto"
        ranking = st.selectbox("Ordenar productos por", list(measures))
        measure = measures[ranking]
        top = products.dropna(subset=[measure]).nlargest(15, measure).copy()
        top["etiqueta"] = top.producto
        if description:
            top["etiqueta"] = (
                top.producto
                + " · "
                + top.descripcion.fillna("Sin descripción").str.slice(0, 55)
            )
        figure = px.bar(
            top.sort_values(measure),
            x=measure,
            y="etiqueta",
            orientation="h",
            hover_data=["producto"],
            labels={measure: ranking, "etiqueta": "Producto"},
        )
        figure.update_yaxes(type="category")
        show_chart(figure)
        st.dataframe(top.drop(columns="etiqueta"), hide_index=True)
        st.caption(
            f"Hasta 15 productos. El ranking comienza por unidades netas: cantidades positivas menos devoluciones o ajustes negativos. Filas sin producto o cantidad: {totals['filas_excluidas']:,}."
        )
        if price:
            st.caption(
                f"Importe neto = cantidad × precio, incluyendo negativos; no es utilidad. Filas sin importe válido: {totals['filas_sin_importe']:,}."
            )
        st.download_button(
            "Descargar resumen de productos",
            products.sort_values(measure, ascending=False)
            .to_csv(index=False)
            .encode("utf-8-sig"),
            "resumen_productos.csv",
            "text/csv",
        )
    elif kind == "Barras por categoría" and categories:
        category = st.selectbox("Categoría", categories)
        metric = st.selectbox("Medida", ["Conteo de registros"] + nums)
        grouped = (
            (
                cleaned.groupby(category).size()
                if metric == "Conteo de registros"
                else cleaned.groupby(category)[metric].sum(min_count=1)
            )
            .dropna()
            .sort_values(ascending=False)
        )
        table = grouped.head(15).rename("valor").reset_index()
        table[category] = table[category].astype(str)
        figure = px.bar(
            table.sort_values("valor"),
            x="valor",
            y=category,
            orientation="h",
            labels={"valor": metric},
        )
        figure.update_yaxes(type="category")
        show_chart(figure)
        st.caption(
            "Hasta 15 categorías con mayor aporte. El porcentaje usa el total de todas las categorías con datos."
        )
        if not grouped.empty and grouped.sum() > 0 and grouped.ge(0).all():
            st.info(
                f"{grouped.index[0]} concentra el {grouped.iloc[0] / grouped.sum():.1%} del total conocido. Revisa esta concentración al planificar."
            )
    elif kind == "Distribución" and nums:
        column = st.selectbox("Variable", nums)
        values = cleaned[column].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(values[np.isfinite(values)], bins=20)
        show_chart(
            go.Figure(
                go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges))
            ).update_layout(xaxis_title=column, yaxis_title="Registros")
        )
    elif kind == "Dispersión" and len(nums) >= 2:
        x = st.selectbox("Eje X", nums)
        y = st.selectbox("Eje Y", nums, index=1)
        sample = cleaned.sample(min(len(cleaned), 5000), random_state=42)
        show_chart(px.scatter(sample, x=x, y=y))
        st.caption(
            "Hasta 5.000 puntos para agilizar la visualización. Una asociación no demuestra causalidad."
        )
    elif kind == "Correlación" and len(nums) >= 2:
        show_chart(
            px.imshow(
                cleaned[nums].corr(min_periods=3),
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
            )
        )
    else:
        st.info(
            "La gráfica necesita columnas categóricas o numéricas compatibles. Revisa las reglas de limpieza."
        )

else:
    date_cols = cleaned.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    nums = cleaned.select_dtypes("number").columns.tolist()
    if not date_cols or not nums:
        st.info(
            "Selecciona una columna de fecha y una numérica en las reglas y pulsa Aplicar limpieza."
        )
        st.stop()
    date = st.selectbox("Fecha", date_cols)
    names = {c.casefold().replace("_", "").replace(" ", ""): c for c in cleaned}
    quantity, price = names.get("quantity"), names.get("unitprice")
    derived = quantity in nums and price in nums
    label = "Importe (Quantity × UnitPrice)"
    metric = st.selectbox("Métrica", ([label] if derived else []) + nums)
    work = cleaned
    if derived and metric == label:
        work = cleaned[[date]].copy()
        work[label] = cleaned[quantity] * cleaned[price]
        st.caption(
            "Incluye movimientos negativos; el importe está en la moneda del archivo y no representa utilidad."
        )
    frequency = st.selectbox("Frecuencia", list(FREQUENCIES))
    method = st.selectbox("Modelo", METHODS)
    horizon = st.slider("Periodos futuros", 1, 12, 4)
    window = st.slider("Periodos recientes para ajustar", 4, 20, 8)
    gaps = st.selectbox(
        "Periodos sin registros",
        ["Tramo continuo más reciente", "Detener", "Cero (cobertura confirmada)"],
    )
    partial = st.checkbox("Excluir periodos límite incompletos", True)
    try:
        series, metadata = aggregate_series(
            work, date, metric, FREQUENCIES[frequency], partial
        )
        result = forecast(series, horizon, method, window, gaps)
    except (ValueError, TypeError) as error:
        st.info(str(error))
        st.stop()
    show_chart(forecast_figure(result, f"Proyección de {metric}"))
    st.dataframe(result["metrics"], hide_index=True)
    st.caption(
        f"Entrenamiento: {result['train_periods']} periodos · Validación: {result['validation_periods']} · Prueba: {result['test_periods']}. La comparación no selecciona automáticamente el modelo."
    )
    st.caption(
        f"Filas válidas: {metadata['filas_usadas']:,} · Sin fecha o métrica: {metadata['filas_sin_par_valido']:,} · Periodos límite excluidos: {metadata['periodos_limite_excluidos']} · Periodos previos al último hueco excluidos: {result['earlier_periods_excluded']}."
    )
    st.warning(
        f"La banda usa errores de validación y es orientativa. Cubrió {result['test_coverage']:.0%} de la prueba. Las fechas proyectadas siguen al histórico del archivo, aunque sea antiguo."
    )
    change = st.slider("Variación del escenario (%)", -30, 50, 0, step=5)
    future = result["future"].copy()
    future["escenario"] = future.prediccion * (1 + change / 100)
    st.metric("Total proyectado", f"{future.prediccion.sum():,.2f}")
    st.dataframe(future, hide_index=True)
    st.caption(
        "El escenario es una hipótesis elegida por ti. Contrasta la proyección con capacidad, costos y existencias para planificar."
    )
    st.download_button(
        "Descargar proyección",
        future.to_csv(index=False).encode("utf-8-sig"),
        "proyeccion.csv",
        "text/csv",
    )
