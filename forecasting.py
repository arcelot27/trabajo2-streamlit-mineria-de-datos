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
    if not pd.api.types.is_datetime64_any_dtype(
        df[date]
    ) or not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            "La fecha debe estar convertida y la métrica debe ser numérica."
        )
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
    series: pd.Series,
    horizon=4,
    method="Promedio reciente",
    window=8,
    missing="Detener",
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
        raise TypeError("La serie debe tener un índice de fechas.")
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
    calibration = validation.to_numpy() - predict(
        train, len(validation), method, window
    )
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
    test_frame = pd.DataFrame(
        {"fecha": test.index, "real": test.values, "prediccion": prediction}
    )
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
    fig.add_trace(
        go.Scatter(x=history.index, y=history.values, mode="lines", name="Histórico")
    )
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
            x=future.fecha,
            y=future.limite_superior_orientativo,
            line_width=0,
            showlegend=False,
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
        title=title,
        xaxis_title="Periodo",
        yaxis_title="Valor de la métrica",
        height=470,
    )
    return fig
