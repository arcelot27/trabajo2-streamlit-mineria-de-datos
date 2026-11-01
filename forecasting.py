"""Proyecciones descriptivas calculadas exclusivamente con el archivo cargado.

Modelos de referencia ligeros con evaluación cronológica, sin datos externos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

METHODS = ["Promedio reciente", "Último periodo", "Tendencia lineal"]
FREQUENCIES = {"Semanal": "W-MON", "Mensual": "MS", "Diaria": "D"}


def aggregate_series(
    df: pd.DataFrame, date: str, metric: str, frequency="W-MON", drop_partial=True
) -> tuple[pd.Series, dict]:
    if date not in df or metric not in df:
        raise ValueError("Selecciona una fecha y una métrica del dataset.")
    if not pd.api.types.is_datetime64_any_dtype(df[date]) or not pd.api.types.is_numeric_dtype(
        df[metric]
    ):
        raise ValueError("La fecha debe estar convertida y la métrica debe ser numérica.")
    work = df[[date, metric]].dropna().sort_values(date)
    if work.empty:
        raise ValueError("No hay pares válidos de fecha y métrica.")
    values = (
        work.set_index(date)[metric]
        .resample(frequency, closed="left", label="left")
        .sum(min_count=1)
    )
    excluded = []
    if drop_partial and len(values) > 2:
        first, last = values.index[0], values.index[-1]
        next_end = last + pd.tseries.frequencies.to_offset(frequency)
        if work[date].min().normalize() > first:
            excluded.append(first)
        if work[date].max().normalize() + pd.Timedelta(days=1) < next_end:
            excluded.append(last)
        values = values.drop(index=excluded)
    return values, {
        "filas_usadas": len(work),
        "filas_sin_par_valido": len(df) - len(work),
        "periodos_limite_excluidos": len(excluded),
        "periodos_sin_datos": int(values.isna().sum()),
    }


def predict(values, horizon: int, method: str, window: int = 8) -> np.ndarray:
    history = np.asarray(values, dtype=float)
    if not len(history) or not np.isfinite(history).all():
        raise ValueError("El modelo requiere una serie sin vacíos ni infinitos.")
    if method == "Último periodo":
        return np.full(horizon, history[-1])
    recent = history[-min(window, len(history)) :]
    if method == "Promedio reciente":
        return np.full(horizon, recent.mean())
    if method == "Tendencia lineal":
        if len(recent) < 2:
            raise ValueError("La tendencia requiere al menos dos periodos.")
        slope, intercept = np.polyfit(np.arange(len(recent), dtype=float), recent, 1)
        return intercept + slope * np.arange(len(recent), len(recent) + horizon)
    raise ValueError("Modelo desconocido.")


def forecast(
    series: pd.Series, horizon=4, method="Promedio reciente", window=8, missing="Detener"
) -> dict:
    if (
        method not in METHODS
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 52
        or not isinstance(window, int)
        or window < 1
    ):
        raise ValueError("Configuración de pronóstico inválida.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("La serie debe tener un índice de fechas.")
    s = series.astype(float).sort_index()
    earlier_periods_excluded = 0
    if s.index.has_duplicates:
        raise ValueError("La serie debe estar agregada con periodos únicos.")
    if missing == "Cero (cobertura confirmada)":
        s = s.fillna(0.0)
    elif missing == "Tramo continuo más reciente" and s.isna().any():
        last_gap = int(np.flatnonzero(s.isna().to_numpy())[-1])
        earlier_periods_excluded = last_gap + 1
        s = s.iloc[last_gap + 1 :]
    elif s.isna().any():
        raise ValueError(
            "Hay periodos sin datos. Confirma cobertura completa para tratarlos como cero o ajusta la frecuencia y los filtros."
        )
    if len(s) < 16:
        raise ValueError(
            "Se requieren al menos 16 periodos para separar entrenamiento, validación y prueba. Usa otra frecuencia o amplía la selección."
        )
    if not np.isfinite(s).all():
        raise ValueError("La serie contiene valores infinitos.")
    frequency = s.index.freq or pd.infer_freq(s.index)
    if frequency is None:
        raise ValueError(
            "La serie requiere periodos regulares. Agrupa los datos antes de predecir."
        )
    # La validación estima una banda orientativa; la prueba final queda separada.
    block = max(3, min(8, len(s) // 5))
    train = s.iloc[: -2 * block]
    validation = s.iloc[-2 * block : -block]
    test = s.iloc[-block:]
    calibration = validation.to_numpy() - predict(train, len(validation), method, window)
    width = float(np.quantile(np.abs(calibration), 0.9))
    fitted_before_test = s.iloc[:-block]
    prediction = predict(fitted_before_test, len(test), method, window)
    rows = []
    for candidate in METHODS:
        candidate_prediction = predict(fitted_before_test, len(test), candidate, window)
        error = test.to_numpy() - candidate_prediction
        denominator = float(np.abs(test).sum())
        rows.append(
            {
                "modelo": candidate,
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "WAPE_pct": float(np.abs(error).sum() / denominator * 100)
                if denominator
                else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    final = predict(s, horizon, method, window)
    offset = pd.tseries.frequencies.to_offset(frequency)
    index = pd.date_range(s.index[-1] + offset, periods=horizon, freq=offset)
    future = pd.DataFrame(
        {
            "fecha": index,
            "prediccion": final,
            "limite_inferior_orientativo": final - width,
            "limite_superior_orientativo": final + width,
        }
    )
    test_frame = pd.DataFrame({"fecha": test.index, "real": test.values, "prediccion": prediction})
    coverage = float((np.abs(test.to_numpy() - prediction) <= width).mean())
    return {
        "history": s,
        "future": future,
        "test": test_frame,
        "metrics": metrics,
        "method": method,
        "validation_periods": len(validation),
        "test_periods": len(test),
        "train_periods": len(train),
        "band_width": width,
        "test_coverage": coverage,
        "earlier_periods_excluded": earlier_periods_excluded,
    }


def forecast_figure(result: dict, title: str) -> go.Figure:
    history, future, test = result["history"], result["future"], result["test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Histórico"))
    fig.add_trace(
        go.Scatter(
            x=test.fecha,
            y=test.prediccion,
            mode="lines+markers",
            name="Predicción del periodo de prueba",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future.fecha, y=future.limite_superior_orientativo, line_width=0, showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future.fecha,
            y=future.limite_inferior_orientativo,
            line_width=0,
            fill="tonexty",
            fillcolor="rgba(56,189,248,.18)",
            name="Rango orientativo",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future.fecha,
            y=future.prediccion,
            mode="lines+markers",
            name="Proyección",
            line_dash="dash",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Periodo", yaxis_title="Valor de la métrica", height=470
    )
    return fig


def render_forecast(df: pd.DataFrame, chart, signature):
    import streamlit as st

    st.subheader("Predicción y decisiones con tu dataset")
    st.caption(
        "Se usa la selección filtrada del archivo cargado. No se consultan datos externos ni se fijan cifras por adelantado."
    )
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    nums = df.select_dtypes("number").columns.tolist()
    if not date_cols or not nums:
        st.info(
            "Para predecir, selecciona una columna de fecha en Limpieza avanzada y convierte la métrica a número en Reglas de limpieza."
        )
        return
    names = {c.casefold().replace("_", "").replace(" ", ""): c for c in df}
    quantity, price = names.get("quantity"), names.get("unitprice")
    derived = bool(quantity in nums and price in nums)
    date = st.selectbox("Fecha para predecir", date_cols)
    derived_label = "Importe (Quantity × UnitPrice)"
    options = ([derived_label] if derived else []) + nums
    metric = st.selectbox("Métrica a predecir", options)
    working = df
    if derived and metric == derived_label:
        working = df[[date]].copy()
        metric = "Importe calculado"
        working[metric] = df[quantity] * df[price]
        st.caption(
            "Importe por línea = Quantity × UnitPrice; incluye devoluciones y ajustes con su signo. Se expresa en la moneda del archivo; no es utilidad."
        )
    frequency = st.selectbox("Frecuencia del pronóstico", list(FREQUENCIES))
    horizon = st.slider("Periodos a proyectar", 1, 12, 4)
    method = st.selectbox("Modelo de referencia", METHODS)
    window = st.slider("Ventana de periodos recientes", 4, 20, 8)
    missing = st.selectbox(
        "Periodos sin observaciones",
        ["Tramo continuo más reciente", "Detener", "Cero (cobertura confirmada)"],
        help="Por defecto se modela desde el último hueco hasta el final, sin rellenar. Solo selecciona cero si el archivo registra todas las operaciones y la ausencia significa que no hubo actividad.",
    )
    drop_partial = st.checkbox("Excluir periodos límite incompletos según fechas disponibles", True)
    try:
        series, metadata = aggregate_series(
            working, date, metric, FREQUENCIES[frequency], drop_partial
        )
        result = forecast(series, horizon, method, window, missing)
    except ValueError as exc:
        st.info(str(exc))
        return
    chart(forecast_figure(result, f"Proyección {frequency.lower()} · {metric}"), "forecast_main")
    if result["earlier_periods_excluded"]:
        st.info(
            f"Se usan {len(result['history'])} periodos consecutivos posteriores al último hueco. Se excluyeron {result['earlier_periods_excluded']} periodos previos para modelar sin inventar actividad en fechas sin registros."
        )
    evaluation = result["metrics"].set_index("modelo").loc[method]
    a, b, c = st.columns(3)
    a.metric("Error absoluto medio de prueba", f"{evaluation.MAE:,.2f}")
    b.metric(
        "Error WAPE de prueba",
        "No definido" if pd.isna(evaluation.WAPE_pct) else f"{evaluation.WAPE_pct:.1f}%",
    )
    c.metric("Total proyectado", f"{result['future'].prediccion.sum():,.2f}")
    st.caption(
        f"Separación cronológica: {result['train_periods']} periodos iniciales de entrenamiento, {result['validation_periods']} de validación de la banda y {result['test_periods']} de prueba. El modelo no usa el tramo de prueba para ajustar sus predicciones de prueba. La proyección final vuelve a ajustar con todo el histórico disponible."
    )
    st.warning(
        f"La banda usa el percentil 90 del error absoluto de validación ({result['validation_periods']} periodos); es orientativa y no garantiza una cobertura del 90%. Cubrió {result['test_coverage']:.0%} de los periodos de prueba. La evaluación es corta y puede variar con otros cortes temporales."
    )
    st.dataframe(result["metrics"], hide_index=True, width="stretch")
    st.caption(
        "La comparación ayuda a evaluar el modelo elegido; no se selecciona automáticamente el mejor sobre la prueba. No se estima estacionalidad anual con un único año de datos."
    )
    st.caption(
        "La proyección comienza después del último periodo utilizado. Si se excluyó un periodo final incompleto, la primera proyección cubre ese periodo. Con un dataset antiguo, las fechas proyectadas también pueden ser históricas respecto al calendario actual."
    )
    st.markdown("### Escenarios para planificar")
    change = st.slider("Variación del escenario sobre la proyección (%)", -30, 50, 0, step=5)
    scenario = result["future"].copy()
    scenario["escenario"] = scenario.prediccion * (1 + change / 100)
    st.caption(
        "La variación es una hipótesis elegida por ti, no un efecto causal estimado ni una recomendación automática."
    )
    st.dataframe(scenario, hide_index=True, width="stretch")
    if evaluation.WAPE_pct > 30:
        st.info(
            "El error de prueba supera el 30% (umbral orientativo). Usa la proyección como escenario y revisa fechas, devoluciones y cambios de demanda antes de comprometer recursos."
        )
    elif pd.notna(evaluation.WAPE_pct):
        st.info(
            "Compara la proyección con tu capacidad disponible y con los periodos de prueba. Confirma costos y existencias antes de decidir compras o contratación."
        )
    if derived:
        st.markdown("### Productos para revisar en la planificación")
        product = names.get("stockcode") or names.get("description")
        if product:
            product_data = df.dropna(subset=[product, quantity]).copy()
            product_data["unidades_positivas"] = product_data[quantity].clip(lower=0)
            product_data["unidades_devueltas_o_ajustadas"] = -product_data[quantity].clip(upper=0)
            product_summary = (
                product_data.groupby(product)
                .agg(
                    unidades_positivas=("unidades_positivas", "sum"),
                    unidades_devueltas_o_ajustadas=("unidades_devueltas_o_ajustadas", "sum"),
                )
                .nlargest(15, "unidades_positivas")
                .reset_index()
            )
            import plotly.express as px

            product_figure = px.bar(
                product_summary,
                x=product,
                y=["unidades_positivas", "unidades_devueltas_o_ajustadas"],
                barmode="group",
            )
            product_figure.update_xaxes(type="category")
            chart(product_figure, "decision_products")
            st.caption(
                "Compara volumen positivo y movimientos negativos para priorizar la revisión de inventario y devoluciones. El dataset no contiene existencias ni costos: no permite calcular un punto de reorden o una utilidad real."
            )
    st.caption(
        f"Filas utilizadas: {metadata['filas_usadas']:,}. Sin par válido: {metadata['filas_sin_par_valido']:,}. Periodos límite excluidos: {metadata['periodos_limite_excluidos']}."
    )
    st.download_button(
        "Descargar proyección y escenario",
        scenario.to_csv(index=False).encode("utf-8-sig"),
        "proyeccion_escenarios.csv",
        "text/csv",
    )
    st.download_button(
        "Descargar evaluación de modelos",
        result["metrics"].to_csv(index=False).encode("utf-8-sig"),
        "evaluacion_prediccion.csv",
        "text/csv",
    )
    st.download_button(
        "Descargar comparación de prueba",
        result["test"].to_csv(index=False).encode("utf-8-sig"),
        "prueba_prediccion.csv",
        "text/csv",
    )
