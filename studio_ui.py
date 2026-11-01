"""Controles para las funciones generales incorporadas a DataClara."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_tools import (
    DATE_FORMATS,
    apply_filters,
    category_totals,
    generate_findings,
    pareto_figure,
    pareto_table,
    time_totals,
)


def advanced_options(raw, tours, signature, numeric, date_defaults=None):
    kwargs = {}
    with st.sidebar.expander("Limpieza avanzada"):
        kwargs["audit_mode"] = st.selectbox(
            "Detalle del registro de limpieza",
            ["Automático", "Resumido", "Detalle completo"],
            help="Automático resume los cambios por columna en archivos de más de 50.000 filas. Detalle completo registra cada celda y consume más memoria.",
        )
        kwargs["drop_empty_rows"] = st.checkbox("Eliminar filas completamente vacías")
        kwargs["duplicate_subset"] = st.multiselect(
            "Clave para detectar duplicados",
            raw.columns.tolist(),
            key=f"dup_{signature}",
            help="Sin selección: compara todas las columnas. Con clave: conserva la primera fila de cada clave.",
        )
        kwargs["missing_columns"] = st.multiselect(
            "Aplicar política de faltantes solo a",
            raw.columns.tolist(),
            key=f"fill_{signature}",
            help="Sin selección: todas las columnas de entrada. Las fechas solo se rellenan si seleccionas Valor anterior.",
        )
        if not tours:
            kwargs["normalize_names"] = st.checkbox("Normalizar nombres de columnas")
            kwargs["drop_empty_columns"] = st.checkbox("Eliminar columnas vacías")
            kwargs["drop_constant_columns"] = st.checkbox("Eliminar columnas constantes")
            if st.checkbox("Eliminar columnas con muchos nulos"):
                kwargs["null_threshold"] = st.slider("Umbral de nulos (%)", 10, 100, 80) / 100
            date_candidates = [c for c in raw if c not in numeric]
            proposed_dates, proposed_format = date_defaults or ([], "Día/mes/año")
            kwargs["date_columns"] = st.multiselect(
                "Columnas de fecha",
                date_candidates,
                default=[c for c in proposed_dates if c in date_candidates],
                key=f"dates_{signature}",
            )
            date_format = st.selectbox(
                "Formato de las fechas",
                list(DATE_FORMATS),
                index=list(DATE_FORMATS).index(proposed_format),
                key=f"date_format_{signature}",
            )
            if proposed_dates:
                st.caption(
                    "Fechas preseleccionadas a partir de formatos reconocibles del archivo. Revisa la convención antes de analizar."
                )
            kwargs["date_format"] = DATE_FORMATS[date_format]
            text_candidates = [
                c
                for c in raw
                if c not in numeric
                and c not in kwargs["date_columns"]
                and not pd.api.types.is_numeric_dtype(raw[c])
                and not pd.api.types.is_datetime64_any_dtype(raw[c])
            ]
            kwargs["text_columns"] = st.multiselect(
                "Columnas para uniformar texto", text_candidates, key=f"text_{signature}"
            )
            kwargs["case_mode"] = st.selectbox(
                "Mayúsculas y minúsculas", ["Conservar", "Minúsculas", "Mayúsculas", "Título"]
            )
            kwargs["remove_accents"] = st.checkbox("Quitar acentos en el texto seleccionado")
        else:
            st.caption(
                "El esquema de giras conserva las columnas necesarias y solo permite marcar atípicos para mantener los indicadores comparables."
            )
        kwargs["outlier_action"] = st.selectbox(
            "Tratamiento de atípicos",
            ["Conservar", "Marcar"] if tours else ["Conservar", "Marcar", "Recortar", "Eliminar"],
        )
        if kwargs["outlier_action"] != "Conservar":
            candidates = [
                c
                for c in raw
                if pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_bool_dtype(raw[c])
            ]
            candidates = list(dict.fromkeys(candidates + numeric))
            if tours:
                candidates = [
                    "recaudacion_usd",
                    "recaudacion_ajustada_usd",
                    "conciertos",
                    "promedio_reportado_usd",
                ]
            kwargs["outlier_columns"] = st.multiselect(
                "Columnas para atípicos", candidates, key=f"outlier_{signature}"
            )
            kwargs["outlier_method"] = st.selectbox("Método de detección", ["IQR", "Z-score"])
            kwargs["outlier_factor"] = st.number_input(
                "Factor de detección",
                min_value=0.1,
                max_value=10.0,
                value=1.5 if kwargs["outlier_method"] == "IQR" else 3.0,
                step=0.1,
                key=f"factor_{kwargs['outlier_method']}",
            )
            st.caption(
                "Los límites se calculan antes de eliminar filas. Recortar modifica los valores; revisa la auditoría. Los atípicos con dispersión cero o menos de 4 datos válidos no se tratan."
            )
    return kwargs


def general_filters(df, signature):
    with st.sidebar.expander("Filtros generales"):
        text = st.text_input("Texto en toda la selección", key=f"filter_text_{signature}")
        keep = st.checkbox("Conservar valores vacíos en los filtros", True)
        cats = [
            c
            for c in df
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and df[c].nunique() <= 200
        ]
        selected_cat = st.selectbox(
            "Filtrar categoría", ["Sin filtro"] + cats, key=f"filter_category_{signature}"
        )
        categories, ranges, dates = {}, {}, {}
        if selected_cat != "Sin filtro":
            options = sorted(df[selected_cat].dropna().astype(str).unique())
            categories[selected_cat] = st.multiselect(
                "Valores permitidos",
                options,
                default=options,
                key=f"filter_values_{signature}_{selected_cat}",
            )
        numeric = df.select_dtypes("number").columns.tolist()
        selected_num = st.selectbox(
            "Filtrar rango numérico", ["Sin filtro"] + numeric, key=f"filter_numeric_{signature}"
        )
        if selected_num != "Sin filtro" and df[selected_num].notna().any():
            low, high = float(df[selected_num].min()), float(df[selected_num].max())
            left, right = st.columns(2)
            minimum = left.number_input(
                "Mínimo", value=low, key=f"range_low_{signature}_{selected_num}"
            )
            maximum = right.number_input(
                "Máximo", value=high, key=f"range_high_{signature}_{selected_num}"
            )
            if minimum > maximum:
                st.warning("El mínimo supera al máximo; la selección quedará vacía.")
            ranges[selected_num] = (minimum, maximum)
        date_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        date_col = st.selectbox(
            "Filtrar fecha", ["Sin filtro"] + date_columns, key=f"filter_date_{signature}"
        )
        if date_col != "Sin filtro" and df[date_col].notna().any():
            bounds = st.date_input(
                "Intervalo de fechas",
                value=(df[date_col].min().date(), df[date_col].max().date()),
                key=f"date_range_{signature}_{date_col}",
            )
            if len(bounds) == 2:
                dates[date_col] = (
                    pd.Timestamp(bounds[0]),
                    pd.Timestamp(bounds[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
                )
            else:
                st.info("Selecciona también la fecha final para aplicar el intervalo.")
        st.caption(
            "Estos filtros se combinan entre sí y con los de giras. Afectan al análisis y a la vista exportada."
        )
    filtered = apply_filters(
        df, text=text, categories=categories, ranges=ranges, dates=dates, keep_missing=keep
    )
    specification = {
        "texto": text,
        "categorias": categories,
        "rangos": ranges,
        "fechas": {c: [str(v) for v in values] for c, values in dates.items()},
        "conservar_vacios": keep,
    }
    return filtered, specification


def render_general_analysis(df, chart, signature):
    st.subheader("Análisis general y hallazgos")
    numeric = df.select_dtypes("number").columns.tolist()
    cats = [
        c
        for c in df
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    dates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    saved = st.session_state.get("business_preferences", {}).get(signature, {})
    default_dimension = saved.get(
        "dimension", "artista" if "artista" in cats else (cats[0] if cats else None)
    )
    dimension = st.selectbox(
        "Dimensión de negocio",
        ["Sin dimensión"] + cats,
        index=cats.index(default_dimension) + 1 if default_dimension in cats else 0,
        key=f"dimension_{signature}",
    )
    dimension = None if dimension == "Sin dimensión" else dimension
    preferred = next(
        (
            c
            for c in numeric
            if any(
                word in c.casefold()
                for word in ["recaudacion_usd", "ingreso", "revenue", "ventas", "importe"]
            )
        ),
        None,
    )
    default_metric = saved.get("metrica", preferred or (numeric[0] if numeric else None))
    selected_metric = st.selectbox(
        "Métrica de negocio",
        ["Conteo de registros"] + numeric,
        index=numeric.index(default_metric) + 1 if default_metric in numeric else 0,
        key=f"business_metric_{signature}",
    )
    metric = None if selected_metric == "Conteo de registros" else selected_metric
    findings = generate_findings(df, dimension, metric)
    for finding in findings:
        st.info(finding)
    st.caption(
        "Hallazgos descriptivos basados en las variables elegidas. Confirma la representatividad de la muestra y la definición de la métrica antes de decidir."
    )
    figures = []
    composition_style = None
    if df.empty:
        return findings, figures, {"dimension": dimension, "metrica": metric}
    if dimension:
        st.markdown("### Aporte y concentración")
        grouped = category_totals(df, dimension, metric)
        fig = px.bar(grouped.head(20), x=dimension, y="valor", labels={"valor": selected_metric})
        chart(fig, "general_categories")
        figures.append(("Aporte por categoría (hasta 20)", fig))
        try:
            pareto = pareto_table(df, dimension, metric)
        except ValueError as exc:
            st.info(str(exc))
            pareto = pd.DataFrame()
        if not pareto.empty:
            fig = pareto_figure(pareto.head(30), dimension)
            chart(fig, "general_pareto")
            st.caption(
                "Hasta 30 categorías. El porcentaje acumulado usa el total de todas las categorías conocidas, aunque no se muestren todas."
            )
            figures.append(("Pareto (hasta 30 categorías)", fig))
            style = st.radio(
                "Composición por categorías",
                ["Dona", "Mapa de árbol"],
                index=1 if saved.get("composicion") == "Mapa de árbol" else 0,
                horizontal=True,
            )
            composition_style = style
            # Para mantener el denominador, agregar las categorías no mostradas.
            composition = grouped.head(10).copy()
            if len(grouped) > 10:
                rest_label = "Otras categorías"
                while rest_label in grouped[dimension].tolist():
                    rest_label += " (resto)"
                composition = pd.concat(
                    [
                        composition,
                        pd.DataFrame(
                            {dimension: [rest_label], "valor": [grouped.iloc[10:]["valor"].sum()]}
                        ),
                    ],
                    ignore_index=True,
                )
            fig = (
                px.pie(composition, names=dimension, values="valor", hole=0.6)
                if style == "Dona"
                else px.treemap(composition, path=[dimension], values="valor")
            )
            chart(fig, "general_composition")
            figures.append(("Composición por categorías", fig))
    temporal_specification = {}
    if dates:
        st.markdown("### Evolución temporal")
        date_col = st.selectbox("Fecha del análisis", dates)
        periods = {
            "Día": "D",
            "Semana (lunes)": "W-MON",
            "Mes": "MS",
            "Trimestre": "QS",
            "Año": "YS",
        }
        frequency_label = st.selectbox("Agrupar por periodo", list(periods), index=2)
        aggregation = (
            st.radio("Agregación temporal", ["Suma", "Promedio"], horizontal=True)
            if metric
            else "Conteo"
        )
        temporal = time_totals(df, date_col, metric, periods[frequency_label], aggregation)
        fig = px.line(
            temporal, x=date_col, y="valor", markers=True, labels={"valor": selected_metric}
        )
        chart(fig, "general_time")
        st.caption(
            "Periodos sin observaciones se muestran como huecos. No se sustituyen por cero. No se infieren causas ni se predice la demanda."
        )
        figures.append(("Evolución temporal", fig))
        temporal_specification = {
            "fecha": date_col,
            "frecuencia": periods[frequency_label],
            "agregacion": aggregation,
        }
    else:
        st.caption(
            "Para análisis temporal, configura una columna de fecha en Limpieza avanzada. Los intervalos de años de giras tienen su gráfica específica en el resumen."
        )
    return (
        findings,
        figures,
        {
            "dimension": dimension,
            "metrica": metric,
            "composicion": composition_style,
            **temporal_specification,
        },
    )
