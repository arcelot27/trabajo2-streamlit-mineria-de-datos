"""Depuración conservadora del archivo de ventas minoristas compartido por el usuario."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from analysis import Options, clean, load_csv


def prepare(source: Path, destination: Path) -> dict:
    with zipfile.ZipFile(source) as archive:
        candidates = [
            i for i in archive.infolist() if Path(i.filename).name.casefold() == "data.csv"
        ]
        if len(candidates) != 1:
            raise ValueError("El ZIP debe contener exactamente un archivo llamado data.csv.")
        if candidates[0].file_size > 50 * 1024 * 1024:
            raise ValueError("data.csv supera el límite de 50 MB.")
        payload = archive.read(candidates[0])
    started = perf_counter()
    raw, read_info = load_csv(payload)
    read_seconds = perf_counter() - started
    required = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    if not required.issubset(raw.columns):
        raise ValueError("No coincide el esquema de ventas minoristas esperado.")
    # Evidencia del formato: hay fechas con segunda componente mayor de 12,
    # lo que obliga a interpretar mes/día/año en este archivo.
    date_parts = raw["InvoiceDate"].str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})")
    month = pd.to_numeric(date_parts[0], errors="coerce")
    day = pd.to_numeric(date_parts[1], errors="coerce")
    evidence = int((day > 12).sum())
    if evidence == 0 or (month > 12).any():
        raise ValueError(
            "No se puede confirmar una convención mes/día consistente; revisa las fechas."
        )
    options = Options(
        numeric_columns=["Quantity", "UnitPrice"],
        date_columns=["InvoiceDate"],
        date_format="%m/%d/%Y %H:%M",
        drop_empty_rows=True,
        drop_empty_columns=True,
        audit_mode="Resumido",
    )
    started = perf_counter()
    cleaned, audit, _ = clean(raw, options)
    clean_seconds = perf_counter() - started
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cleaned.to_csv(
        destination / "retail_limpio.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    cleaned.to_parquet(destination / "retail_limpio.parquet", index=False, compression="snappy")
    audit.to_csv(destination / "retail_auditoria.csv", index=False, encoding="utf-8-sig")
    flags = pd.DataFrame(
        {
            "fila_dataset_limpio": cleaned.index + 1,
            "factura_cancelacion": cleaned.InvoiceNo.str.upper().str.startswith("C", na=False),
            "cantidad_no_positiva": cleaned.Quantity.le(0).fillna(False),
            "precio_no_positivo": cleaned.UnitPrice.le(0).fillna(False),
            "cliente_ausente": cleaned.CustomerID.isna(),
            "descripcion_ausente": cleaned.Description.isna(),
        }
    )
    flag_columns = flags.columns[1:]
    flags.loc[flags[flag_columns].any(axis=1)].to_parquet(
        destination / "retail_casos_revision.parquet", index=False
    )
    summary = {
        "fuente": str(source),
        "sha256_csv_original": hashlib.sha256(payload).hexdigest(),
        "lectura": read_info,
        "filas_originales": len(raw),
        "filas_limpias": len(cleaned),
        "filas_eliminadas": len(raw) - len(cleaned),
        "celdas_vacias_por_columna": cleaned.isna().sum().astype(int).to_dict(),
        "fecha_minima": str(cleaned.InvoiceDate.min()),
        "fecha_maxima": str(cleaned.InvoiceDate.max()),
        "evidencia_mes_dia": evidence,
        "casos_para_revision": flags[flag_columns].sum().astype(int).to_dict(),
        "tiempos_segundos": {
            "lectura": read_seconds,
            "limpieza": clean_seconds,
            "exportacion": perf_counter() - started,
        },
        "opciones": asdict(options),
        "parquet_bytes": (destination / "retail_limpio.parquet").stat().st_size,
        "csv_bytes": (destination / "retail_limpio.csv").stat().st_size,
    }
    (destination / "retail_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "LEEME.md").write_text(
        "# Dataset de ventas depurado\n\n"
        f"Filas originales: {len(raw):,}. Filas conservadas: {len(cleaned):,}. Eliminadas: {len(raw) - len(cleaned):,}.\n\n"
        "Se normalizaron espacios y nulos escritos como texto, se convirtieron cantidades y precios y se interpretaron fechas mes/día/año con hora. "
        "Se eliminaron duplicados exactos después de limpiar. InvoiceNo, StockCode y CustomerID se conservaron como texto. No se imputaron clientes ni descripciones.\n\n"
        "Se conservaron devoluciones, cancelaciones y precios no positivos. retail_casos_revision.parquet identifica estos casos y los faltantes con filas referidas al dataset limpio (primera fila de datos = 1). "
        "Son señales para revisión, no pruebas de error. Los conteos se superponen.\n\n"
        "Para cargar más rápido usa retail_limpio.parquet. Ambos formatos contienen todas las filas conservadas, sin muestreo. "
        "La auditoría es resumida por columna y operación; cantidad indica cuántas celdas o filas afecta cada operación. "
        "El ZIP original no se modifica. Tiempos y configuración están en retail_resumen.json.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "datasets")
    args = parser.parse_args()
    summary = prepare(args.entrada, args.salida)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
