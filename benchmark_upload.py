"""Mide el flujo real de carga de un CSV del ZIP con AppTest, sin navegador."""

import argparse
import json
import zipfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from analysis import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--salida", type=Path, default=Path("rendimiento_carga.json"))
    args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as archive:
        item = next(i for i in archive.infolist() if i.filename.endswith("data.csv"))
        payload = archive.read(item)
    upload = SimpleNamespace(name="data.csv", getvalue=lambda: payload)
    at = AppTest.from_file(Path(__file__).with_name("app.py"), default_timeout=180).run()
    with (
        patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=upload),
        patch("analysis.clean", wraps=clean) as cleaner,
    ):
        started = perf_counter()
        at.run()
        first = perf_counter() - started
        assert not at.exception, [e.message for e in at.exception]
        assert cleaner.call_count == 1
        first_charts = len(at.get("plotly_chart"))
        started = perf_counter()
        at.run()
        cached = perf_counter() - started
        assert not at.exception
        assert cleaner.call_count == 1, "La limpieza se repitió innecesariamente"
        started = perf_counter()
        at.session_state["active_section"] = "Predicción y decisiones"
        at.run()
        prediction = perf_counter() - started
        assert not at.exception, [e.message for e in at.exception]
        assert cleaner.call_count == 1
        assert any(m.label == "Error WAPE de prueba" for m in at.metric), [i.value for i in at.info]
        summary = {
            "archivo": "data.csv",
            "filas_originales": len(at.session_state["loaded_dataset"][0]),
            "filas_limpias": len(at.session_state["cleaning_result"][0]),
            "auditoria_registros": len(at.session_state["cleaning_result"][1]),
            "limpieza_segundos": at.session_state["cleaning_seconds"],
            "primera_carga_interfaz_segundos": first,
            "recarga_cache_segundos": cached,
            "abrir_prediccion_segundos": prediction,
            "graficas_en_resumen": first_charts,
            "graficas_en_prediccion": len(at.get("plotly_chart")),
            "ejecuciones_limpieza": cleaner.call_count,
        }
    args.salida.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
