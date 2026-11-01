"""Procesamiento por lotes con el mismo motor usado por Streamlit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px

from analysis import Options, clean, is_tours, normalize
from data_tools import (
    build_excel,
    build_html_report,
    generate_findings,
    load_dataset,
    pareto_figure,
    pareto_table,
    profile,
    suggested_types,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Limpia y normaliza CSV, Excel, JSON y Parquet con DataClara."
    )
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--salida", type=Path, default=Path("resultados_lote"))
    parser.add_argument(
        "--numericas",
        nargs="*",
        default=None,
        help="Sin esta opción se usan sugerencias conservadoras.",
    )
    parser.add_argument("--fechas", nargs="*", default=[])
    parser.add_argument(
        "--formato-fecha", default="%d/%m/%Y", help="Ej.: %%d/%%m/%%Y, ISO8601 o mixed_dayfirst"
    )
    parser.add_argument("--decimal", choices=[".", ","], default=".")
    parser.add_argument("--normalizacion", choices=["minmax", "zscore"], default="minmax")
    parser.add_argument(
        "--hoja", default=None, help="Nombre de la hoja Excel; por defecto la primera."
    )
    args = parser.parse_args(argv)
    try:
        raw, info = load_dataset(args.entrada.read_bytes(), args.entrada.name, sheet=args.hoja or 0)
        hints = suggested_types(raw, args.decimal)
        numeric = (
            args.numericas
            if args.numericas is not None
            else hints.loc[hints.sugerencia == "Número", "columna"].tolist()
        )
        options = Options(
            numeric_columns=[] if is_tours(raw) else numeric,
            decimal=args.decimal,
            date_columns=args.fechas,
            date_format=args.formato_fecha,
        )
        cleaned, audit, tours = clean(raw, options)
        cols = [
            c
            for c in cleaned.select_dtypes("number")
            if c
            not in {
                "posicion",
                "mejor_posicion",
                "mejor_posicion_historica",
                "anio_inicio",
                "anio_fin",
            }
        ]
        method = "Min–Max (0 a 1)" if args.normalizacion == "minmax" else "Z-score"
        normalized, parameters = normalize(cleaned, cols, method)
        metric = "recaudacion_usd" if tours else (cols[0] if cols else None)
        cats = [
            c
            for c in cleaned
            if not pd.api.types.is_numeric_dtype(cleaned[c])
            and not pd.api.types.is_datetime64_any_dtype(cleaned[c])
        ]
        dim = "artista" if tours else (cats[0] if cats else None)
        findings = generate_findings(cleaned, dim, metric)
        figures = []
        if metric:
            figures.append(
                ("Distribución", px.histogram(cleaned, x=metric, template="plotly_dark"))
            )
        if dim:
            try:
                pareto = pareto_table(cleaned, dim, metric)
                if not pareto.empty:
                    figures.append(
                        ("Concentración por categoría", pareto_figure(pareto.head(30), dim))
                    )
            except ValueError:
                pass
        report = (
            f"Fuente: {args.entrada.name}\nLectura: {info}\nFilas originales: {len(raw)}\n"
            f"Filas limpias: {len(cleaned)}\nConfiguración: {json.dumps(asdict(options), ensure_ascii=False)}\n"
            f"Normalización: {method}; columnas: {', '.join(cols)}. Ajustada sobre todo el dataset limpio.\n"
            "Las celdas faltantes permanecen vacías. Los indicadores conservan sus unidades. Las asociaciones no prueban causalidad."
        )
        if tours:
            report += "\nMuestra seleccionada de giras. Recaudación no equivale a rentabilidad. Ajuste a USD de 2022 tomado del archivo sin recalcular inflación."
        frames = {
            "Datos limpios": cleaned,
            "Datos normalizados": normalized,
            "Parámetros": parameters,
            "Registro limpieza": audit,
            "Perfil": profile(cleaned),
            "Configuración": pd.DataFrame({"informe": [report]}),
        }
        excel = None
        try:
            excel = build_excel(frames)
        except ValueError as exc:
            print(f"Excel omitido: {exc}")
        html = build_html_report(args.entrada.name, report, cleaned, audit, findings, figures)
        args.salida.mkdir(parents=True, exist_ok=True)
        for name, frame in [
            ("datos_limpios.csv", cleaned),
            ("datos_normalizados.csv", normalized),
            ("parametros_normalizacion.csv", parameters),
            ("registro_limpieza.csv", audit),
        ]:
            frame.to_csv(args.salida / name, index=False, encoding="utf-8-sig")
        if excel is not None:
            (args.salida / "informe_dataclara.xlsx").write_bytes(excel)
        (args.salida / "informe_dataclara.html").write_text(html, encoding="utf-8")
        cleaned.to_parquet(args.salida / "datos_limpios.parquet", index=False)
        cleaned.to_json(
            args.salida / "datos_limpios.json",
            orient="records",
            date_format="iso",
            force_ascii=False,
        )
        print(
            f"Procesados {len(raw)} registros; conservados {len(cleaned)}. Entregables en {args.salida.resolve()}"
        )
        return 0
    except (ValueError, OSError, TypeError, ImportError) as exc:
        parser.exit(1, f"No se pudo procesar: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
