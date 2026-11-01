"""Verificación de funciones incorporadas tras comparar el proyecto de referencia."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import plotly.express as px
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from analysis import Options, clean, number, parse_dates
from data_tools import (
    apply_filters,
    build_excel,
    build_html_report,
    diagnostics,
    excel_sheets,
    generate_findings,
    load_dataset,
    pareto_table,
    quality_indicators,
    suggested_types,
    time_totals,
)
from procesar import main as process_cli

ROOT = Path(__file__).resolve().parent


def start_app():
    at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    next(w for w in at.radio if w.label == "Fuente de datos").set_value(
        "Dataset de la carpeta"
    ).run()
    return at


class StudioEngineTests(unittest.TestCase):
    def test_excel_sheets_and_identifiers(self):
        data = io.BytesIO()
        with pd.ExcelWriter(data, engine="openpyxl") as writer:
            pd.DataFrame({"codigo": ["001", "002"], "valor": [5, 10]}).to_excel(
                writer, sheet_name="Ventas", index=False
            )
            pd.DataFrame({"otra": [9]}).to_excel(writer, sheet_name="Otra", index=False)
        self.assertEqual(excel_sheets(data.getvalue()), ["Ventas", "Otra"])
        raw, _ = load_dataset(data.getvalue(), "datos.xlsx", sheet="Ventas")
        result, _, _ = clean(raw, Options(numeric_columns=["valor"]))
        self.assertEqual(result.codigo.tolist(), ["001", "002"])
        self.assertEqual(result.valor.sum(), 15)
        duplicate = io.BytesIO()
        pd.DataFrame([[1, 2]], columns=["A", "a"]).to_excel(duplicate, index=False)
        with self.assertRaises(ValueError):
            load_dataset(duplicate.getvalue(), "mal.xlsx")

    def test_json_parquet_and_invalid_inputs(self):
        payload = json.dumps(
            [{"codigo": "001", "valor": 10.0, "detalle": {"zona": "Norte"}}]
        ).encode()
        raw, _ = load_dataset(payload, "datos.json")
        self.assertEqual(raw.iloc[0]["codigo"], "001")
        self.assertIn("detalle.zona", raw)
        result, _, _ = clean(raw, Options())
        self.assertEqual(result.valor.iloc[0], 10)
        buffer = io.BytesIO()
        result.to_parquet(buffer, index=False)
        again, _ = load_dataset(buffer.getvalue(), "datos.parquet")
        self.assertEqual(again.codigo.iloc[0], "001")
        lines, _ = load_dataset(b'{"a":1}\n{"a":2}', "datos.jsonl")
        self.assertEqual(len(lines), 2)
        for filename, data in [
            ("datos.exe", b"abc"),
            ("datos.json", b"[]"),
            ("datos.json", b"[1,2]"),
            ("datos.csv", b"a,b\n1,2\n3,4,5"),
        ]:
            with self.assertRaises(ValueError):
                load_dataset(data, filename)

    def test_text_names_missing_and_date_audit(self):
        raw = pd.DataFrame(
            {
                "Región": [" NORTE ", "norte", "sin dato"],
                "Region": ["A", "A", "B"],
                "fecha": ["2024-01-05", "05/02/2024", "31/02/2024"],
            }
        )
        result, audit, _ = clean(
            raw,
            Options(
                text_columns=["Región"],
                case_mode="Minúsculas",
                normalize_names=True,
                date_columns=["fecha"],
                date_format="mixed_dayfirst",
            ),
        )
        self.assertEqual(result.columns.tolist(), ["region", "region_2", "fecha"])
        self.assertEqual(result.region.iloc[0], "norte")
        self.assertTrue(pd.isna(result.region.iloc[2]))
        self.assertEqual(result.fecha.iloc[0], pd.Timestamp("2024-01-05"))
        self.assertEqual(result.fecha.iloc[1], pd.Timestamp("2024-02-05"))
        self.assertTrue(pd.isna(result.fecha.iloc[2]))
        self.assertTrue(audit.motivo.str.contains("Fecha inválida").any())
        self.assertEqual(raw.iloc[0, 0], " NORTE ")

    def test_missing_strategies_and_column_removal(self):
        raw = pd.DataFrame(
            {"grupo": ["A", None, "A"], "valor": [1.0, None, 5.0], "vacia": [None] * 3}
        )
        result, audit, _ = clean(
            raw,
            Options(
                missing="Media / Moda", missing_columns=["valor", "grupo"], drop_empty_columns=True
            ),
        )
        self.assertEqual(result.valor.tolist(), [1.0, 3.0, 5.0])
        self.assertEqual(result.grupo.tolist(), ["A"] * 3)
        self.assertNotIn("vacia", result)
        self.assertTrue(audit.motivo.str.contains("Columna vacía").any())
        ordered, _, _ = clean(
            pd.DataFrame({"fecha": pd.to_datetime([None, "2024-01-01", None])}),
            Options(deduplicate=False, missing="Valor anterior"),
        )
        self.assertTrue(pd.isna(ordered.fecha.iloc[0]))
        self.assertEqual(ordered.fecha.iloc[2], pd.Timestamp("2024-01-01"))
        boolean, _, _ = clean(
            pd.DataFrame({"b": pd.Series([True, None], dtype="boolean")}),
            Options(missing="Mediana / Sin dato"),
        )
        self.assertTrue(pd.isna(boolean.b.iloc[1]))

    def test_outliers_fixed_bounds_and_key_duplicates(self):
        raw = pd.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E", "F"],
                "x": [1.0, 1.0, 2.0, 2.0, 3.0, 100.0],
                "y": [1.0, 2.0, 2.0, 3.0, 200.0, 3.0],
            }
        )
        flagged, audit, _ = clean(raw, Options(outlier_columns=["x"], outlier_action="Marcar"))
        self.assertTrue(flagged.x_atipico.iloc[-1])
        self.assertEqual(flagged.x.iloc[-1], 100)
        clipped, _, _ = clean(raw, Options(outlier_columns=["x"], outlier_action="Recortar"))
        self.assertLess(clipped.x.max(), 100)
        first, _, _ = clean(raw, Options(outlier_columns=["x", "y"], outlier_action="Eliminar"))
        second, _, _ = clean(raw, Options(outlier_columns=["y", "x"], outlier_action="Eliminar"))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 4)
        duplicate, _, _ = clean(
            pd.DataFrame({"id": ["A", "A"], "valor": [1, 2]}), Options(duplicate_subset=["id"])
        )
        self.assertEqual(duplicate.valor.tolist(), [1])

    def test_suggestions_diagnostics_and_quality(self):
        raw = pd.DataFrame(
            {"codigo": ["001", "002"], "valor": ["10", "20"], "region": [" norte ", "NORTE"]}
        )
        hints = suggested_types(raw).set_index("columna")
        self.assertEqual(hints.loc["codigo", "sugerencia"], "Texto")
        self.assertEqual(hints.loc["valor", "sugerencia"], "Número")
        self.assertTrue(diagnostics(raw).problema.str.contains("Espacios").any())
        for frame in [raw, pd.DataFrame(), pd.DataFrame({"a": [None, None]})]:
            self.assertTrue(0 <= quality_indicators(frame)["puntaje"] <= 100)
        self.assertEqual(number("45%"), 45)
        self.assertEqual(number("(1,234.50)"), -1234.5)

    def test_filters_two_category_findings_and_pareto(self):
        df = pd.DataFrame(
            {
                "grupo": ["A", "B", "A", None],
                "valor": [1.0, 2.0, 7.0, None],
                "fecha": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", None]),
            }
        )
        view = apply_filters(
            df, categories={"grupo": ["A"]}, ranges={"valor": (2, 10)}, keep_missing=False
        )
        self.assertEqual(view.valor.tolist(), [7])
        self.assertTrue(generate_findings(df, "grupo", "valor"))
        pareto = pareto_table(df, "grupo", "valor")
        self.assertEqual(pareto.grupo.tolist(), ["A", "B"])
        self.assertEqual(pareto.acumulado_pct.iloc[-1], 100)
        df.loc[0, "valor"] = -1
        with self.assertRaises(ValueError):
            pareto_table(df, "grupo", "valor")

    def test_time_gaps_and_dates(self):
        df = pd.DataFrame(
            {"fecha": pd.to_datetime(["2024-01-01", "2024-03-01"]), "valor": [10.0, 20.0]}
        )
        temporal = time_totals(df, "fecha", "valor")
        self.assertEqual(len(temporal), 3)
        self.assertTrue(pd.isna(temporal.valor.iloc[1]))
        self.assertEqual(temporal.valor.iloc[2], 20)
        parsed = parse_dates(pd.Series(["01/05/2024"]), "%d/%m/%Y")
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-05-01"))

    def test_excel_and_html_exports(self):
        df = pd.DataFrame({"texto": ["=1+1", "<script>alert(1)</script>"], "valor": [1, 2]})
        workbook = load_workbook(io.BytesIO(build_excel({"Datos": df})), data_only=False)
        self.assertEqual(workbook["Datos"]["A2"].data_type, "s")
        self.assertEqual(workbook["Datos"].freeze_panes, "A2")
        fig = px.bar(df, x="texto", y="valor", template="plotly_dark")
        report = build_html_report(
            "<test>", "config", df, pd.DataFrame(), ["<b>dato</b>"], [("Gráfica", fig)]
        )
        self.assertIn("&lt;test&gt;", report)
        self.assertIn("&lt;b&gt;dato&lt;/b&gt;", report)
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertIn("plotly.js", report)

    def test_batch_processor(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "datos.csv"
            source.write_text("grupo;valor\nA;10\nB;20\n", encoding="utf-8")
            self.assertEqual(process_cli([str(source), "--salida", str(folder / "salida")]), 0)
            out = folder / "salida"
            self.assertTrue((out / "informe_dataclara.xlsx").exists())
            scaled = pd.read_csv(out / "datos_normalizados.csv")
            self.assertEqual(scaled.valor.tolist(), [0.0, 1.0])


class StudioAppTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch("data_tools.local_datasets", return_value=[ROOT / "my_file (1).csv"])
        )

    def test_uploaded_csv_displays_data_and_reuses_cleaning(self):
        payload = b"CustomerID,Description,Quantity,UnitPrice\n001, PRODUCT A ,2,12.50\n002,PRODUCT B,3,8.00\n"
        uploaded = SimpleNamespace(name="ventas_subidas.csv", getvalue=lambda: payload)
        at = start_app()
        source = next(w for w in at.radio if w.label == "Fuente de datos")
        with (
            patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=uploaded),
            patch("analysis.clean", wraps=clean) as cleaner,
        ):
            source.set_value("Subir un archivo").run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Archivo recibido" in w.value for w in at.success))
            self.assertEqual(len(at.tabs), 8)
            preview = at.dataframe[0].value
            self.assertEqual(preview.CustomerID.tolist(), ["001", "002"])
            self.assertEqual(at.session_state["cleaning_result"][0].Quantity.tolist(), [2.0, 3.0])
            self.assertEqual(cleaner.call_count, 1)
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(cleaner.call_count, 1)
            next(w for w in at.checkbox if w.label == "Normalizar espacios").uncheck().run()
            self.assertFalse(at.exception)
            self.assertEqual(cleaner.call_count, 2)
            at.session_state["active_section"] = "Descargas"
            at.run()
            next(w for w in at.button if w.label == "Preparar paquete completo (.zip)").click()
            at.session_state["active_section"] = "Descargas"
            at.run()
            self.assertFalse(at.exception)
            with zipfile.ZipFile(io.BytesIO(at.session_state["zip_export"][2])) as bundle:
                self.assertIn("datos_limpios.csv", bundle.namelist())

    def test_demo_advanced_dates_and_exports(self):
        at = start_app()
        next(w for w in at.radio if w.label == "Fuente de datos").set_value(
            "Demostración de ventas"
        ).run()
        self.assertFalse(at.exception)
        next(w for w in at.multiselect if w.label == "Columnas de fecha").set_value(["Fecha"]).run()
        next(w for w in at.selectbox if w.label == "Formato de las fechas").set_value(
            "Mixto (día/mes primero, ISO conservado)"
        ).run()
        self.assertFalse(at.exception)
        at.session_state["active_section"] = "Análisis general"
        at.run()
        self.assertTrue(any(w.label == "Fecha del análisis" for w in at.selectbox))
        next(w for w in at.selectbox if w.label == "Dimensión de negocio").set_value("Region")
        at.session_state["active_section"] = "Análisis general"
        at.run()
        next(w for w in at.selectbox if w.label == "Métrica de negocio").set_value("Ingreso")
        at.session_state["active_section"] = "Análisis general"
        at.run()
        next(w for w in at.radio if w.label == "Composición por categorías").set_value(
            "Mapa de árbol"
        )
        at.session_state["active_section"] = "Análisis general"
        at.run()
        self.assertFalse(at.exception)
        at.session_state["active_section"] = "Descargas"
        at.run()
        next(w for w in at.button if w.label == "Generar Excel e informe HTML").click()
        at.session_state["active_section"] = "Descargas"
        at.run()
        self.assertFalse(at.exception)
        self.assertIn("excel", at.session_state["studio_exports"])
        next(w for w in at.selectbox if w.label == "Filtrar categoría").set_value("Region").run()
        next(w for w in at.multiselect if w.label == "Valores permitidos").set_value([]).run()
        next(
            w for w in at.checkbox if w.label == "Conservar valores vacíos en los filtros"
        ).uncheck().run()
        self.assertFalse(at.exception)
        at.session_state["active_section"] = "Resumen ejecutivo"
        at.run()
        self.assertTrue(any("No hay registros" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
