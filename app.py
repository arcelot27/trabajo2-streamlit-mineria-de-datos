"""Interfaz esencial para limpiar, analizar y proyectar datos del usuario."""

import hashlib
import io
from functools import partial
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
from i18n import translate

language = st.session_state.get("language", "es")
t = partial(translate, language=language)


def toggle_language():
    st.session_state.language = (
        "en" if st.session_state.get("language", "es") == "es" else "es"
    )
    # Send the selected values again so their displayed option labels are refreshed.
    for key in st.session_state:
        if key.startswith("localized_"):
            value = st.session_state[key]
            if translate(value, language="en") != value:
                st.session_state[key] = value


def translated_table(frame, *value_columns):
    labels = {col: frame[col].map(t, na_action="ignore") for col in value_columns}
    return frame.assign(**labels).rename(columns=t)


st.set_page_config(
    page_title=t("DataClara | Minería de datos"),
    page_icon=":material/analytics:",
    layout="wide",
)
st.sidebar.button(
    "Traducir · English" if language == "es" else "Translate · Español",
    key="toggle_language",
    on_click=toggle_language,
)
st.title(t("DataClara · Minería de datos"))
st.caption(t("Carga tu archivo, revisa la limpieza y analiza sus resultados."))
uploaded = st.sidebar.file_uploader(
    t("Dataset CSV o Excel"), type=["csv", "xlsx"], key="dataset_upload"
)
if uploaded is None:
    st.info(t("Sube un dataset desde el panel lateral para comenzar."))
    st.stop()

payload = uploaded.getvalue()
sheet, separator = 0, "Automático"
try:
    if uploaded.name.lower().endswith(".xlsx"):
        with pd.ExcelFile(io.BytesIO(payload)) as book:
            sheet = st.sidebar.selectbox(
                t("Hoja de Excel"), book.sheet_names, key="excel_sheet"
            )
    else:
        separator = st.sidebar.selectbox(
            t("Separador CSV"),
            ["Automático", ",", ";", "\t", "|"],
            key="localized_csv_separator",
            format_func=t,
        )
    signature = hashlib.sha256(payload + str((sheet, separator)).encode()).hexdigest()
    if (
        st.session_state.get("source") != signature
        or "date_suggestions" not in st.session_state
    ):
        with st.spinner(t("Leyendo dataset…")):
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
    message, delimiter, details = str(error).partition(": ")
    st.error(
        t("No se pudo leer el archivo: {error}", error=t(message) + delimiter + details)
    )
    st.stop()

st.sidebar.caption(
    t(
        "{rows:,} filas · {columns} columnas",
        rows=len(original),
        columns=len(original.columns),
    )
)
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
    st.subheader(t("Reglas de limpieza"))
    decimal = st.selectbox(
        t("Separador decimal"), [".", ","], key=f"decimal_{signature}"
    )
    numeric = st.multiselect(
        t("Columnas numéricas"),
        original.columns,
        default=st.session_state.numeric_suggestions,
        key=f"numeric_columns_{signature}",
    )
    proposed_dates, proposed_format = st.session_state.date_suggestions
    dates = st.multiselect(
        t("Columnas de fecha"),
        original.columns,
        default=proposed_dates,
        key=f"date_columns_{signature}",
    )
    date_format_key = f"localized_date_format_{signature}"
    date_format = st.selectbox(
        t("Formato de fechas"),
        list(formats),
        index=None
        if date_format_key in st.session_state
        else list(formats.values()).index(proposed_format),
        key=date_format_key,
        format_func=t,
    )
    if proposed_dates:
        st.caption(
            t(
                "Fechas preseleccionadas a partir de una muestra distribuida del archivo. Revisa el formato antes de aplicar cambios."
            )
        )
    st.caption(
        t(
            "Confirma el orden de día y mes. Los identificadores deben conservarse como texto."
        )
    )
    deduplicate = st.checkbox(
        t("Eliminar duplicados"), True, key=f"deduplicate_{signature}"
    )
    lowercase = st.checkbox(
        t("Convertir texto a minúsculas"), key=f"lowercase_{signature}"
    )
    missing = st.selectbox(
        t("Valores faltantes"),
        ["Conservar", "Mediana / Sin dato", "Eliminar filas"],
        key=f"localized_missing_values_{signature}",
        format_func=t,
    )
    remove_null = st.checkbox(
        t("Eliminar columnas con muchos nulos"), key=f"remove_null_columns_{signature}"
    )
    threshold = st.slider(
        t("Umbral de nulos (%)"), 10, 100, 80, key=f"null_threshold_{signature}"
    )
    outliers = st.selectbox(
        t("Valores atípicos"),
        ["Conservar", "Recortar (IQR)"],
        key=f"localized_outliers_{signature}",
        format_func=t,
    )
    submitted = st.form_submit_button(
        t("Aplicar limpieza"), type="primary", key=f"apply_cleaning_{signature}"
    )

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
        st.error(t("Una columna no puede ser numérica y fecha a la vez."))
        st.stop()
    with st.spinner(t("Limpiando dataset…")):
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
    t(
        "{file} · {rows:,} filas conservadas · Limpieza: {seconds:.2f} s",
        file=uploaded.name,
        rows=len(cleaned),
        seconds=st.session_state.seconds,
    )
)
section = st.sidebar.radio(
    t("Sección"),
    ["Limpieza", "Normalización", "Análisis", "Predicción"],
    key="localized_section",
    format_func=t,
)


def show_chart(figure):
    st.plotly_chart(figure, width="stretch")


if section == "Limpieza":
    a, b, c = st.columns(3)
    a.metric(t("Filas originales"), len(original))
    b.metric(t("Filas eliminadas"), len(original) - len(cleaned))
    c.metric(t("Faltantes después"), int(cleaned.isna().sum().sum()))
    before, after = st.columns(2)
    with before:
        st.subheader(t("Antes"))
        st.dataframe(original.head(100), hide_index=True)
        st.dataframe(translated_table(quality(original)), hide_index=True)
    with after:
        st.subheader(t("Después"))
        st.dataframe(cleaned.head(100), hide_index=True)
        st.dataframe(translated_table(quality(cleaned)), hide_index=True)
    st.caption(
        t(
            "Vista previa de hasta 100 filas. El original se conserva; la descarga contiene todas las filas limpias."
        )
    )
    st.subheader(t("Registro de limpieza"))
    st.dataframe(translated_table(audit, "operacion"), hide_index=True)
    st.caption(
        t(
            "El registro resume cantidades por operación. Las conversiones inválidas quedan vacías; los atípicos se conservan salvo que elijas recortarlos."
        )
    )
    if st.button(t("Preparar CSV limpio")):
        st.session_state.csv = cleaned.to_csv(index=False).encode("utf-8-sig")
    if "csv" in st.session_state:
        st.download_button(
            t("Descargar dataset limpio"),
            st.session_state.csv,
            "datos_limpios.csv",
            "text/csv",
        )

elif section == "Normalización":
    columns = st.multiselect(
        t("Variables numéricas a normalizar"),
        cleaned.select_dtypes("number").columns,
        key="normalize_columns",
    )
    method = st.selectbox(
        t("Método"),
        ["Min–Max (0 a 1)", "Z-score"],
        key="localized_normalize_method",
        format_func=t,
    )
    st.caption(
        t(
            "Se cambia la escala en una copia: Min–Max lleva los valores a [0, 1]; Z-score usa media y desviación. Los faltantes se conservan."
        )
    )
    if columns:
        key = (tuple(columns), method)
        if st.session_state.get("normalization_key") != key:
            st.session_state.normalized = normalize(cleaned, columns, method)
            st.session_state.normalization_key = key
        normalized, parameters = st.session_state.normalized
        st.dataframe(normalized.head(100), hide_index=True)
        st.dataframe(translated_table(parameters, "metodo", "estado"), hide_index=True)
        if st.button(t("Preparar CSV normalizado")):
            st.download_button(
                t("Descargar dataset normalizado"),
                normalized.to_csv(index=False).encode("utf-8-sig"),
                "datos_normalizados.csv",
                "text/csv",
                on_click="ignore",
            )

elif section == "Análisis":
    if cleaned.empty or not len(cleaned.columns):
        st.info(t("No hay datos con estas reglas. Ajusta la limpieza."))
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
    kind = st.selectbox(t("Gráfica"), kinds, key="localized_chart", format_func=t)
    if kind == "Productos más vendidos" and nums:
        with st.expander(
            t("Columnas para el resumen de productos"),
            expanded=not bool(quantity and product),
        ):
            product = st.selectbox(
                t("Columna de producto o código"),
                cleaned.columns,
                index=cleaned.columns.get_loc(product) if product else 0,
                key="product_column",
            )
            quantity = st.selectbox(
                t("Columna de cantidad"),
                nums,
                index=nums.index(quantity) if quantity in nums else 0,
                key="quantity_column",
            )
            optional = ["Sin seleccionar"] + cleaned.columns.tolist()
            description = (
                names.get("description")
                or names.get("descripcion")
                or names.get("descripción")
            )
            description = st.selectbox(
                t("Descripción del producto (opcional)"),
                optional,
                index=None
                if st.session_state.get("localized_product_description") in optional
                else optional.index(description)
                if description
                else 0,
                key="localized_product_description",
                format_func=lambda value: (
                    t(value) if value == "Sin seleccionar" else value
                ),
            )
            prices = ["Sin seleccionar"] + nums
            price = (
                names.get("unitprice")
                or names.get("preciounitario")
                or names.get("precio")
            )
            price = st.selectbox(
                t("Precio por unidad (opcional)"),
                prices,
                index=None
                if st.session_state.get("localized_unit_price") in prices
                else prices.index(price)
                if price in nums
                else 0,
                key="localized_unit_price",
                format_func=lambda value: (
                    t(value) if value == "Sin seleccionar" else value
                ),
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
                t(
                    "No hay filas con producto y cantidad válidos. Revisa las columnas seleccionadas y la limpieza."
                )
            )
            st.stop()
        st.subheader(t("Ventas por producto"))
        st.caption(t("Resumen de todo el dataset limpio cargado."))
        a, b, c, d = st.columns(4)
        a.metric(t("Productos diferentes"), len(products))
        b.metric(t("Unidades netas"), f"{totals['unidades_netas']:,.2f}")
        c.metric(t("Unidades positivas"), f"{totals['unidades_vendidas']:,.2f}")
        d.metric(
            t("Unidades devueltas o ajustadas"), f"{totals['unidades_devueltas']:,.2f}"
        )
        measures = {
            "Unidades netas": "unidades_netas",
            "Unidades positivas (antes de devoluciones)": "unidades_vendidas",
            "Unidades devueltas o ajustadas": "unidades_devueltas",
        }
        if price:
            value = totals["importe_neto"]
            st.metric(
                t("Importe neto registrado"),
                f"{value:,.2f}" if pd.notna(value) else t("Sin datos"),
            )
            measures["Importe neto"] = "importe_neto"
        ranking = st.selectbox(
            t("Ordenar productos por"),
            list(measures),
            key="localized_product_ranking",
            format_func=t,
        )
        measure = measures[ranking]
        top = products.dropna(subset=[measure]).nlargest(15, measure).copy()
        top["etiqueta"] = top.producto
        if description:
            top["etiqueta"] = (
                top.producto
                + " · "
                + top.descripcion.fillna(t("Sin descripción")).str.slice(0, 55)
            )
        figure = px.bar(
            top.sort_values(measure),
            x=measure,
            y="etiqueta",
            orientation="h",
            hover_data=["producto"],
            labels={
                measure: t(ranking),
                "etiqueta": t("Producto"),
                "producto": t("producto"),
            },
        )
        figure.update_yaxes(type="category")
        show_chart(figure)
        st.dataframe(translated_table(top.drop(columns="etiqueta")), hide_index=True)
        st.caption(
            t(
                "Hasta 15 productos. El ranking comienza por unidades netas: cantidades positivas menos devoluciones o ajustes negativos. Filas sin producto o cantidad: {rows:,}.",
                rows=totals["filas_excluidas"],
            )
        )
        if price:
            st.caption(
                t(
                    "Importe neto = cantidad × precio, incluyendo negativos; no es utilidad. Filas sin importe válido: {rows:,}.",
                    rows=totals["filas_sin_importe"],
                )
            )
        st.download_button(
            t("Descargar resumen de productos"),
            products.sort_values(measure, ascending=False)
            .to_csv(index=False)
            .encode("utf-8-sig"),
            "resumen_productos.csv",
            "text/csv",
        )
    elif kind == "Barras por categoría" and categories:
        category = st.selectbox(t("Categoría"), categories, key="category")
        metric = st.selectbox(
            t("Medida"),
            ["Conteo de registros"] + nums,
            key="localized_category_measure",
            format_func=lambda value: (
                t(value) if value == "Conteo de registros" else value
            ),
        )
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
            labels={"valor": t(metric) if metric == "Conteo de registros" else metric},
        )
        figure.update_yaxes(type="category")
        show_chart(figure)
        st.caption(
            t(
                "Hasta 15 categorías con mayor aporte. El porcentaje usa el total de todas las categorías con datos."
            )
        )
        if not grouped.empty and grouped.sum() > 0 and grouped.ge(0).all():
            st.info(
                t(
                    "{category} concentra el {share:.1%} del total conocido. Revisa esta concentración al planificar.",
                    category=grouped.index[0],
                    share=grouped.iloc[0] / grouped.sum(),
                )
            )
    elif kind == "Distribución" and nums:
        column = st.selectbox(t("Variable"), nums, key="distribution_column")
        values = cleaned[column].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(values[np.isfinite(values)], bins=20)
        show_chart(
            go.Figure(
                go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges))
            ).update_layout(xaxis_title=column, yaxis_title=t("Registros"))
        )
    elif kind == "Dispersión" and len(nums) >= 2:
        x = st.selectbox(t("Eje X"), nums, key="scatter_x")
        y = st.selectbox(t("Eje Y"), nums, index=1, key="scatter_y")
        sample = cleaned.sample(min(len(cleaned), 5000), random_state=42)
        show_chart(px.scatter(sample, x=x, y=y))
        st.caption(
            t(
                "Hasta 5.000 puntos para agilizar la visualización. Una asociación no demuestra causalidad."
            )
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
            t(
                "La gráfica necesita columnas categóricas o numéricas compatibles. Revisa las reglas de limpieza."
            )
        )

else:
    date_cols = cleaned.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    nums = cleaned.select_dtypes("number").columns.tolist()
    if not date_cols or not nums:
        st.info(
            t(
                "Selecciona una columna de fecha y una numérica en las reglas y pulsa Aplicar limpieza."
            )
        )
        st.stop()
    date = st.selectbox(t("Fecha"), date_cols, key="forecast_date")
    names = {c.casefold().replace("_", "").replace(" ", ""): c for c in cleaned}
    quantity, price = names.get("quantity"), names.get("unitprice")
    derived = quantity in nums and price in nums
    label = "Importe (Quantity × UnitPrice)"
    metric = st.selectbox(
        t("Métrica"),
        ([label] if derived else []) + nums,
        key="localized_forecast_metric",
        format_func=lambda value: (
            t(value) if value == "Importe (Quantity × UnitPrice)" else value
        ),
    )
    work = cleaned
    if derived and metric == label:
        work = cleaned[[date]].copy()
        work[label] = cleaned[quantity] * cleaned[price]
        st.caption(
            t(
                "Incluye movimientos negativos; el importe está en la moneda del archivo y no representa utilidad."
            )
        )
    frequency = st.selectbox(
        t("Frecuencia"),
        list(FREQUENCIES),
        key="localized_forecast_frequency",
        format_func=t,
    )
    method = st.selectbox(
        t("Modelo"), METHODS, key="localized_forecast_model", format_func=t
    )
    horizon = st.slider(t("Periodos futuros"), 1, 12, 4, key="forecast_horizon")
    window = st.slider(
        t("Periodos recientes para ajustar"), 4, 20, 8, key="forecast_window"
    )
    gaps = st.selectbox(
        t("Periodos sin registros"),
        ["Tramo continuo más reciente", "Detener", "Cero (cobertura confirmada)"],
        key="localized_forecast_gaps",
        format_func=t,
    )
    drop_partial = st.checkbox(
        t("Excluir periodos límite incompletos"), True, key="forecast_partial"
    )
    try:
        series, metadata = aggregate_series(
            work, date, metric, FREQUENCIES[frequency], drop_partial
        )
        result = forecast(series, horizon, method, window, gaps)
    except (ValueError, TypeError) as error:
        st.info(t(str(error)))
        st.stop()
    metric_label = t(metric) if metric == label else metric
    show_chart(
        forecast_figure(
            result, t("Proyección de {metric}", metric=metric_label), language=language
        )
    )
    st.dataframe(translated_table(result["metrics"], "modelo"), hide_index=True)
    st.caption(
        t(
            "Entrenamiento: {train} periodos · Validación: {validation} · Prueba: {test}. La comparación no selecciona automáticamente el modelo.",
            train=result["train_periods"],
            validation=result["validation_periods"],
            test=result["test_periods"],
        )
    )
    st.caption(
        t(
            "Filas válidas: {used:,} · Sin fecha o métrica: {missing:,} · Periodos límite excluidos: {partial} · Periodos previos al último hueco excluidos: {earlier}.",
            used=metadata["filas_usadas"],
            missing=metadata["filas_sin_par_valido"],
            partial=metadata["periodos_limite_excluidos"],
            earlier=result["earlier_periods_excluded"],
        )
    )
    st.warning(
        t(
            "La banda usa errores de validación y es orientativa. Cubrió {coverage:.0%} de la prueba. Las fechas proyectadas siguen al histórico del archivo, aunque sea antiguo.",
            coverage=result["test_coverage"],
        )
    )
    change = st.slider(
        t("Variación del escenario (%)"), -30, 50, 0, step=5, key="scenario_adjustment"
    )
    future = result["future"].copy()
    future["escenario"] = future.prediccion * (1 + change / 100)
    st.metric(t("Total proyectado"), f"{future.prediccion.sum():,.2f}")
    st.dataframe(translated_table(future), hide_index=True)
    st.caption(
        t(
            "El escenario es una hipótesis elegida por ti. Contrasta la proyección con capacidad, costos y existencias para planificar."
        )
    )
    st.download_button(
        t("Descargar proyección"),
        future.to_csv(index=False).encode("utf-8-sig"),
        "proyeccion.csv",
        "text/csv",
    )
