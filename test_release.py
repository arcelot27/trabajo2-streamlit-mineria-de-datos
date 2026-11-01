"""Regresiones detectadas al preparar la aplicación para publicación."""

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis import load_csv
from data_tools import build_excel, build_html_report
from forecasting import forecast

ROOT = Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    def test_csv_rejects_implicit_index_and_truncated_fields(self):
        for payload in [b"a,b\n1,2,3\n4,5,6\n", b"a,b\n1,2,3\n"]:
            with self.assertRaises(ValueError):
                load_csv(payload)

    def test_excel_limit_runs_before_allocating_workbook(self):
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with patch("data_tools.MAX_EXCEL_CELLS", 5), patch("pandas.ExcelWriter") as writer:
            with self.assertRaisesRegex(ValueError, "CSV o Parquet"):
                build_excel({"Datos": frame})
            writer.assert_not_called()

    def test_html_limits_audit_and_keeps_complete_download_note(self):
        audit = pd.DataFrame({"motivo": ["VISIBLE", "OCULTO"]})
        with patch("data_tools.MAX_HTML_AUDIT_ROWS", 1):
            document = build_html_report("archivo", "config", pd.DataFrame({"a": [1]}), audit, [])
        self.assertIn("VISIBLE", document)
        self.assertNotIn("OCULTO", document)
        self.assertIn("CSV de auditoría", document)

    def test_forecast_rejects_irregular_periods_and_non_dates(self):
        regular = pd.date_range("2024-01-01", periods=25, freq="W-MON")
        irregular = regular.delete(10)
        with self.assertRaisesRegex(ValueError, "regulares"):
            forecast(pd.Series(np.ones(len(irregular)), index=irregular))
        with self.assertRaisesRegex(ValueError, "fechas"):
            forecast(pd.Series(np.ones(25)))

    def test_new_upload_discards_previous_heavy_exports(self):
        first = SimpleNamespace(name="primero.csv", getvalue=lambda: b"grupo,valor\nA,10\nB,20")
        second = SimpleNamespace(name="segundo.csv", getvalue=lambda: b"grupo,valor\nC,30\nD,40")
        at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
        with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=first):
            at.run()
            self.assertFalse(at.exception)
            for key in ["studio_exports", "zip_export", "normalization_result"]:
                at.session_state[key] = b"exportacion anterior"
        with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=second):
            at.run()
            self.assertFalse(at.exception)
            for key in ["studio_exports", "zip_export", "normalization_result"]:
                self.assertNotIn(key, at.session_state)
            self.assertEqual(at.session_state["cleaning_result"][0].valor.tolist(), [30.0, 40.0])

    def test_large_excel_does_not_block_html_or_parquet(self):
        upload = SimpleNamespace(name="archivo.csv", getvalue=lambda: b"grupo,valor\nA,10\nB,20")
        at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
        with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=upload):
            at.run()
            at.session_state["active_section"] = "Descargas"
            at.run()
            next(b for b in at.button if b.label == "Generar Excel e informe HTML").click()
            at.session_state["active_section"] = "Descargas"
            with patch("data_tools.MAX_EXCEL_CELLS", 1):
                at.run()
            self.assertFalse(at.exception)
            exports = at.session_state["studio_exports"]
            self.assertIsNone(exports["excel"])
            self.assertTrue(exports["html"])
            self.assertEqual(
                pd.read_parquet(io.BytesIO(exports["parquet"])).valor.tolist(), [10.0, 20.0]
            )


if __name__ == "__main__":
    unittest.main()
