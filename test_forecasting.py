"""Predicciones sin fuga temporal, auditoría resumida y carga bajo demanda."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis import Options, clean
from data_tools import infer_date_settings, suggested_types
from forecasting import aggregate_series, forecast


class ForecastTests(unittest.TestCase):
    def test_forecasts_and_temporal_isolation(self):
        series = pd.Series(
            np.arange(50, dtype=float) * 10 + 100,
            index=pd.date_range("2024-01-01", periods=50, freq="W-MON"),
        )
        result = forecast(series, horizon=4, method="Tendencia lineal", window=8)
        self.assertAlmostEqual(
            result["metrics"].set_index("modelo").loc["Tendencia lineal", "MAE"], 0, places=8
        )
        self.assertAlmostEqual(result["future"].prediccion.iloc[0], 600)
        changed = series.copy()
        changed.iloc[-result["test_periods"] :] += 10000
        changed_result = forecast(changed, method="Tendencia lineal", window=8)
        np.testing.assert_allclose(result["test"].prediccion, changed_result["test"].prediccion)
        self.assertAlmostEqual(result["band_width"], changed_result["band_width"])
        self.assertGreater(changed_result["metrics"].iloc[0].MAE, result["metrics"].iloc[0].MAE)

    def test_missing_short_and_zero_series(self):
        series = pd.Series(
            np.zeros(25), index=pd.date_range("2024-01-01", periods=25, freq="W-MON")
        )
        result = forecast(series)
        self.assertTrue(result["metrics"].WAPE_pct.isna().all())
        self.assertTrue((result["future"].prediccion == 0).all())
        series.iloc[5] = np.nan
        with self.assertRaises(ValueError):
            forecast(series)
        self.assertEqual(len(forecast(series, missing="Cero (cobertura confirmada)")["future"]), 4)
        with self.assertRaises(ValueError):
            forecast(series.head(10))

    def test_aggregate_preserves_return_sign_and_excludes_boundaries(self):
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(["2024-01-03", "2024-01-08", "2024-01-09", "2024-01-16"]),
                "importe": [10.0, 30.0, -5.0, 20.0],
            }
        )
        series, meta = aggregate_series(df, "fecha", "importe")
        self.assertEqual(series.tolist(), [25.0])
        self.assertEqual(meta["periodos_limite_excluidos"], 2)

    def test_summary_audit_matches_detailed_data_and_counts(self):
        raw = pd.DataFrame(
            {
                "id": ["01", "02", "02", "03"],
                "valor": ["10", "20", "20", "malo"],
                "texto": [" A ", "B", "B", None],
            }
        )
        detail, detail_audit, _ = clean(
            raw, Options(numeric_columns=["valor"], audit_mode="Detalle completo")
        )
        summarized, summary_audit, _ = clean(
            raw, Options(numeric_columns=["valor"], audit_mode="Resumido")
        )
        pd.testing.assert_frame_equal(detail, summarized)
        self.assertEqual(detail_audit.cantidad.sum(), summary_audit.cantidad.sum())
        self.assertLess(len(summary_audit), len(detail_audit))

    def test_identifier_protection_for_arbitrary_uploaded_data(self):
        raw = pd.DataFrame(
            {
                "InvoiceNo": ["536365", "C536366"],
                "StockCode": ["0123", "12345"],
                "CustomerID": ["001", "002"],
                "Quantity": ["10", "-5"],
            }
        )
        types = suggested_types(raw).set_index("columna").sugerencia
        self.assertTrue((types[["InvoiceNo", "StockCode", "CustomerID"]] == "Texto").all())
        self.assertEqual(types.Quantity, "Número")
        raw["InvoiceDate"] = ["12/1/2010 8:26", "12/20/2010 9:12"]
        self.assertEqual(
            suggested_types(raw).set_index("columna").loc["InvoiceDate", "sugerencia"],
            "Fecha sugerida",
        )

    def test_date_inference_uses_file_values(self):
        raw = pd.DataFrame({"fecha_operacion": ["12/1/2010 8:26", "12/20/2010 9:12"]})
        cols, fmt = infer_date_settings(raw, suggested_types(raw))
        self.assertEqual(cols, ["fecha_operacion"])
        self.assertEqual(fmt, "Mes/día/año con hora")
        raw = pd.DataFrame({"otra_fecha": ["01/02/2024", "02/03/2024"]})
        cols, _ = infer_date_settings(raw, suggested_types(raw))
        self.assertEqual(cols, [])


class ForecastAppTests(unittest.TestCase):
    def test_upload_only_initial_screen_and_lazy_predictions(self):
        root = Path(__file__).resolve().parent
        at = AppTest.from_file(root / "app.py", default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.metric)
        self.assertFalse(at.get("plotly_chart"))
        rows = [f"2024-{month:02}-01,{month * 10},2" for month in range(1, 13)]
        rows += [f"2025-{month:02}-01,{month * 12},3" for month in range(1, 13)]
        payload = ("fecha,unidades,precio\n" + "\n".join(rows)).encode()
        upload = SimpleNamespace(name="datos_nuevos.csv", getvalue=lambda: payload)
        with (
            patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=upload),
            patch("analysis.clean", wraps=clean) as cleaner,
        ):
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.get("plotly_chart")), 1)
            self.assertEqual(cleaner.call_count, 1)
            next(w for w in at.multiselect if w.label == "Columnas de fecha").set_value(
                ["fecha"]
            ).run()
            next(w for w in at.selectbox if w.label == "Formato de las fechas").set_value(
                "Año-mes-día"
            ).run()
            at.session_state["active_section"] = "Predicción y decisiones"
            at.run()
            next(w for w in at.selectbox if w.label == "Frecuencia del pronóstico").set_value(
                "Mensual"
            )
            at.session_state["active_section"] = "Predicción y decisiones"
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.get("plotly_chart")), 1)
            self.assertTrue(any(m.label == "Error WAPE de prueba" for m in at.metric))
            calls = cleaner.call_count
            at.session_state["active_section"] = "Datos"
            at.run()
            self.assertEqual(cleaner.call_count, calls)
            self.assertFalse(at.get("plotly_chart"))


if __name__ == "__main__":
    unittest.main()
