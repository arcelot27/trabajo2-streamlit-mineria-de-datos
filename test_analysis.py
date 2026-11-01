"""Pruebas de las decisiones de limpieza y de los flujos principales."""

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis import Options, artist_summary, clean, load_csv, normalize, number

ROOT = Path(__file__).resolve().parent


def start_app():
    at = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    next(w for w in at.radio if w.label == "Fuente de datos").set_value(
        "Dataset de la carpeta"
    ).run()
    return at


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.raw, _ = load_csv((ROOT / "my_file (1).csv").read_bytes())

    def test_real_dataset_and_ambiguous_money(self):
        df, audit, tours = clean(self.raw, Options())
        self.assertTrue(tours)
        self.assertEqual(len(df), 20)
        self.assertEqual(df["recaudacion_usd"].sum(), 5759018065)
        cher = df.loc[df["artista"] == "Cher"].iloc[0]
        self.assertTrue(pd.isna(cher["promedio_reportado_usd"]))
        self.assertAlmostEqual(cher["ingreso_por_concierto_usd"], 200000000 / 325)
        self.assertEqual(df.iloc[0]["anio_fin"], 2024)
        self.assertEqual(df.iloc[2]["mejor_posicion"], 1)
        self.assertTrue(audit["motivo"].str.contains("inválido").any())
        self.assertEqual(self.raw.iloc[13]["Average gross"], "$ 615,39")

    def test_locale_and_invalid_numbers(self):
        self.assertEqual(number("$229,100,000[b]"), 229100000)
        self.assertTrue(pd.isna(number("$ 615,39")))
        self.assertEqual(number("1.234,50", ","), 1234.5)
        self.assertTrue(pd.isna(number("abc")))
        self.assertTrue(pd.isna(number("1e309")))

    def test_generic_duplicate_and_missing_policies(self):
        raw, _ = load_csv(b"categoria;valor\n A ;1,000\nA;1,000\nB;\nC;invalido\n")
        df, audit, tours = clean(raw, Options(numeric_columns=["valor"]))
        self.assertFalse(tours)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["valor"].isna().sum(), 2)
        self.assertTrue(audit["motivo"].str.contains("Duplicado").any())
        filled, _, _ = clean(raw, Options(numeric_columns=["valor"], missing="Mediana / Sin dato"))
        self.assertEqual(filled["valor"].tolist(), [1000, 1000, 1000])
        dropped, _, _ = clean(
            raw, Options(numeric_columns=["valor"], missing="Eliminar filas incompletas")
        )
        self.assertEqual(len(dropped), 1)

    def test_weighted_average_uses_only_complete_pairs(self):
        df = pd.DataFrame(
            {
                "artista": ["A", "A"],
                "gira": ["X", "Y"],
                "recaudacion_usd": [100.0, 900.0],
                "conciertos": [10.0, None],
            }
        )
        self.assertEqual(artist_summary(df, "recaudacion_usd").iloc[0]["por_concierto"], 10)
        df["conciertos"] = float("nan")
        self.assertTrue(pd.isna(artist_summary(df, "recaudacion_usd").iloc[0]["por_concierto"]))

    def test_invalid_files(self):
        for data in [b"", b"a,a\n1,2", b"a,b\n", b"a;\n1;2"]:
            with self.assertRaises(ValueError):
                load_csv(data)

    def test_normalization_real_dataset(self):
        df, _, _ = clean(self.raw, Options())
        scaled, parameters = normalize(
            df, ["recaudacion_usd", "conciertos", "promedio_reportado_usd"]
        )
        self.assertAlmostEqual(scaled["recaudacion_usd"].min(), 0)
        self.assertAlmostEqual(scaled["recaudacion_usd"].max(), 1)
        self.assertEqual(scaled["promedio_reportado_usd"].isna().sum(), 1)
        self.assertEqual(parameters.iloc[0]["minimo"], 150000000)
        self.assertEqual(scaled["artista"].tolist(), df["artista"].tolist())
        self.assertEqual(df["recaudacion_usd"].max(), 780000000)
        z, _ = normalize(df, ["recaudacion_usd"], "Z-score")
        self.assertAlmostEqual(z["recaudacion_usd"].mean(), 0)
        self.assertAlmostEqual(z["recaudacion_usd"].std(ddof=0), 1)

    def test_normalization_constant_missing_and_invalid_columns(self):
        df = pd.DataFrame(
            {
                "constante": [5.0, 5.0, None],
                "vacia": [float("nan")] * 3,
                "texto": ["A", "B", "C"],
                "bandera": [True, False, True],
            }
        )
        for method in ["Min–Max (0 a 1)", "Z-score"]:
            scaled, parameters = normalize(df, ["constante", "vacia"], method)
            self.assertEqual(scaled["constante"].dropna().tolist(), [0, 0])
            self.assertTrue(scaled["vacia"].isna().all())
            self.assertTrue(pd.isna(scaled.iloc[2]["constante"]))
            self.assertEqual(parameters.iloc[0]["validos"], 2)
        for col in ["texto", "bandera", "inexistente"]:
            with self.assertRaises(ValueError):
                normalize(df, [col])


class AppTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch("data_tools.local_datasets", return_value=[ROOT / "my_file (1).csv"])
        )

    def test_normalization_controls_preserve_business_metrics(self):
        at = start_app()
        self.assertFalse(at.exception)
        original_total = next(m.value for m in at.metric if m.label == "Recaudación de la muestra")
        at.session_state["active_section"] = "Normalización"
        at.run()
        next(w for w in at.radio if w.label == "Método de normalización").set_value("Z-score")
        at.session_state["active_section"] = "Normalización"
        at.run()
        self.assertFalse(at.exception)
        next(w for w in at.multiselect if w.label == "Columnas a normalizar").set_value([])
        at.session_state["active_section"] = "Normalización"
        at.run()
        self.assertFalse(at.exception)
        at.session_state["active_section"] = "Resumen ejecutivo"
        at.run()
        self.assertEqual(
            next(m.value for m in at.metric if m.label == "Recaudación de la muestra"),
            original_total,
        )

    def test_generic_csv_and_empty_cleaned_result(self):
        raw, _ = load_csv(b"categoria;valor\nA;100\nB;\n")
        with patch("analysis.load_csv", return_value=(raw, "CSV de prueba")):
            at = start_app()
            self.assertFalse(at.exception)
            next(
                w for w in at.multiselect if w.label == "Columnas que deben ser numéricas"
            ).set_value(["valor"]).run()
            self.assertFalse(at.exception)
            next(w for w in at.selectbox if w.label == "Valores faltantes").set_value(
                "Mediana / Sin dato"
            ).run()
            self.assertFalse(at.exception)
        empty_raw = self._incomplete_tours()
        with patch("analysis.load_csv", return_value=(empty_raw, "CSV de prueba")):
            at = start_app()
            next(w for w in at.selectbox if w.label == "Valores faltantes").set_value(
                "Eliminar filas incompletas"
            ).run()
            self.assertFalse(at.exception)

    def _incomplete_tours(self):
        raw, _ = load_csv((ROOT / "my_file (1).csv").read_bytes())
        raw["Artist"] = pd.NA
        return raw

    def test_dashboard_filters_and_rules(self):
        at = start_app()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("plotly_chart")), 4)
        artist_widget = next(w for w in at.multiselect if w.label == "Artistas")
        artist_widget.set_value(["Taylor Swift"]).run()
        self.assertFalse(at.exception)
        self.assertEqual(next(m.value for m in at.metric if m.label == "Artistas"), "1")
        artist_widget.set_value([]).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("No hay registros" in i.value for i in at.info))

    def test_adjusted_basis_and_explorer(self):
        at = start_app()
        next(w for w in at.radio if w.label == "Base monetaria").set_value(
            "Ajustada (USD de 2022)"
        ).run()
        self.assertFalse(at.exception)
        at.session_state["active_section"] = "Explorador"
        at.run()
        for kind in ["Distribución", "Dispersión", "Caja y atípicos", "Correlación"]:
            next(w for w in at.selectbox if w.label == "Tipo de gráfica").set_value(kind)
            at.session_state["active_section"] = "Explorador"
            at.run()
            self.assertFalse(at.exception, kind)
        next(w for w in at.selectbox if w.label == "Valores faltantes").set_value(
            "Eliminar filas incompletas"
        ).run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
