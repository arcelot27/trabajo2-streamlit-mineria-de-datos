"""Lectura de archivos, perfilado, hallazgos y exportaciones de DataClara."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import NULL_TOKENS, column_names, load_csv, number, quality

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".parquet", ".pq"}
MAX_EXCEL_CELLS = 1_000_000
MAX_HTML_AUDIT_ROWS = 5_000


def local_datasets(root: Path) -> list[Path]:
    """Lista archivos locales compatibles sin recorrer carpetas de resultados."""
    candidates = [*root.iterdir(), *(root / "datasets").glob("*")]
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED),
        key=lambda path: (path.parent != root, path.name),
    )


DATE_FORMATS = {
    "Día/mes/año": "%d/%m/%Y",
    "Mes/día/año": "%m/%d/%Y",
    "Día/mes/año con hora": "%d/%m/%Y %H:%M",
    "Mes/día/año con hora": "%m/%d/%Y %H:%M",
    "Año-mes-día": "%Y-%m-%d",
    "ISO 8601 (con hora opcional)": "ISO8601",
    "Mixto (día/mes primero, ISO conservado)": "mixed_dayfirst",
}


def excel_sheets(data: bytes) -> list[str]:
    try:
        with pd.ExcelFile(io.BytesIO(data)) as book:
            return book.sheet_names
    except ImportError as exc:
        raise ValueError("Falta el lector de Excel. Instala requirements.txt.") from exc
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el libro Excel: {exc}") from exc


def load_dataset(
    data: bytes, filename: str, separator="Automático", encoding="Automático", sheet=0
) -> tuple[pd.DataFrame, str]:
    if not data or len(data) > 50 * 1024 * 1024:
        raise ValueError("El archivo debe contener datos y pesar menos de 50 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Formato no soportado. Usa CSV, TSV, Excel, JSON, JSONL o Parquet.")
    if suffix in {".csv", ".tsv", ".txt"}:
        return load_csv(
            data, "\t" if suffix == ".tsv" and separator == "Automático" else separator, encoding
        )
    try:
        if suffix in {".xlsx", ".xls"}:
            table = pd.read_excel(
                io.BytesIO(data), sheet_name=sheet, header=None, dtype=object, keep_default_na=False
            )
            if table.empty:
                raise ValueError("La hoja está vacía.")
            headers = [str(c).strip() for c in table.iloc[0]]
            if any(not c for c in headers) or len(set(c.casefold() for c in headers)) != len(
                headers
            ):
                raise ValueError("Los encabezados deben ser únicos y no estar vacíos.")
            frame = table.iloc[1:].copy()
            frame.columns = headers
            info = f"Excel · hoja {sheet}"
        elif suffix in {".json", ".jsonl"}:
            # No se infieren identificadores numéricos de cadenas JSON.
            text = data.decode("utf-8-sig")
            if suffix == ".jsonl":
                payload = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                payload = json.loads(text)
            if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
                frame = pd.json_normalize(payload)
            elif isinstance(payload, dict) and all(isinstance(v, list) for v in payload.values()):
                frame = pd.DataFrame(payload)
            else:
                raise ValueError(
                    "JSON debe ser una lista de registros o un objeto de columnas con listas."
                )
            for col in frame:
                frame[col] = frame[col].map(
                    lambda v: (
                        json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    )
                )
            info = "JSON · registros tabulares"
        else:
            frame = pd.read_parquet(io.BytesIO(data))
            info = "Parquet"
    except ImportError as exc:
        raise ValueError(
            "Falta una dependencia para este formato. Instala requirements.txt."
        ) from exc
    except Exception as exc:
        raise ValueError(f"No se pudo leer {Path(filename).name}: {exc}") from exc
    if frame.empty or not len(frame.columns):
        raise ValueError("El archivo no contiene registros tabulares.")
    frame.columns = [str(c) for c in frame.columns]
    normalized_names = [c.strip().casefold() for c in frame.columns]
    if any(not c for c in normalized_names) or len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Los encabezados deben ser únicos y no estar vacíos.")
    # Conservar filas/columnas vacías para que su eliminación sea auditable.
    for col in frame.select_dtypes(include=["object", "string"]):
        frame[col] = frame[col].map(lambda v: pd.NA if isinstance(v, str) and not v.strip() else v)
    return frame.reset_index(drop=True), info


def suggested_types(df: pd.DataFrame, decimal=".") -> pd.DataFrame:
    rows = []
    for col in df:
        s = df[col].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            suggestion, why = "Fecha", "Tipo de fecha del archivo"
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            suggestion, why = "Número", "Tipo numérico del archivo"
        else:
            values = s.astype(str)
            values = values.loc[~values.str.strip().str.lower().isin(NULL_TOKENS)]
            label = column_names([col])[col]
            protected = bool(
                set(label.split("_")).intersection(
                    {
                        "id",
                        "codigo",
                        "code",
                        "cod",
                        "sku",
                        "telefono",
                        "nit",
                        "documento",
                        "postal",
                        "ref",
                        "factura",
                        "invoice",
                    }
                )
            )
            if set(label.split("_")).intersection({"fecha", "date", "time", "timestamp"}):
                protected = False
            protected = protected or bool(values.str.fullmatch(r"0\d+").any())
            if protected:
                suggestion, why = "Texto", "Posible identificador; conservar ceros iniciales"
            elif (
                not values.empty
                and values.map({v: pd.notna(number(v, decimal)) for v in values.unique()}).mean()
                >= 0.9
            ):
                suggestion, why = (
                    "Número",
                    "Al menos 90% de valores válidos con la convención seleccionada",
                )
            elif (
                not values.empty
                and values.str.match(r"^(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})").mean() >= 0.8
            ):
                suggestion, why = (
                    "Fecha sugerida",
                    "Confirma el orden de día y mes antes de convertir",
                )
            else:
                suggestion, why = "Texto", "Sin conversión inequívoca"
        rows.append({"columna": col, "sugerencia": suggestion, "motivo": why})
    return pd.DataFrame(rows, columns=["columna", "sugerencia", "motivo"])


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        rows.append(
            {"columna": "(dataset)", "problema": "Filas duplicadas", "cantidad": duplicate_count}
        )
    for col in df:
        s = df[col]
        if s.isna().any():
            rows.append(
                {"columna": col, "problema": "Valores vacíos", "cantidad": int(s.isna().sum())}
            )
        if s.nunique(dropna=False) <= 1:
            rows.append(
                {"columna": col, "problema": "Columna vacía o constante", "cantidad": len(s)}
            )
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            text = s.dropna().astype(str)
            placeholders = text.str.strip().str.lower().isin(NULL_TOKENS)
            if placeholders.any():
                rows.append(
                    {
                        "columna": col,
                        "problema": "Nulos escritos como texto",
                        "cantidad": int(placeholders.sum()),
                    }
                )
            normalized = text.str.replace(r"\s+", " ", regex=True).str.strip()
            if (text != normalized).any():
                rows.append(
                    {
                        "columna": col,
                        "problema": "Espacios inconsistentes",
                        "cantidad": int((text != normalized).sum()),
                    }
                )
            variants = int(normalized.nunique() - normalized.str.casefold().nunique())
            if variants:
                rows.append(
                    {
                        "columna": col,
                        "problema": "Variantes por mayúsculas/minúsculas",
                        "cantidad": variants,
                    }
                )
    return pd.DataFrame(rows, columns=["columna", "problema", "cantidad"])


def infer_date_settings(df: pd.DataFrame, suggestions: pd.DataFrame) -> tuple[list[str], str]:
    """Sugiere conversiones solo si las cadenas permiten confirmar la convención."""
    selected, formats = [], []
    for col in suggestions.loc[suggestions.sugerencia == "Fecha sugerida", "columna"]:
        values = pd.Series(df[col].dropna().astype(str).unique())
        if values.empty:
            continue
        values = values.sample(min(2000, len(values)), random_state=42)
        iso = values.str.match(r"^\d{4}-\d{2}-\d{2}")
        if iso.all():
            selected.append(col)
            formats.append("ISO 8601 (con hora opcional)")
            continue
        parts = values.loc[~iso].str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})(.*)$")
        a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        with_time = parts[3].fillna("").str.match(r"^\s+\d{1,2}:\d{2}$").all()
        if parts[0].notna().mean() < 0.9:
            continue
        if (a > 12).any() and not (b > 12).any():
            fmt = (
                "Mixto (día/mes primero, ISO conservado)"
                if iso.any()
                else ("Día/mes/año con hora" if with_time else "Día/mes/año")
            )
        elif (b > 12).any() and not (a > 12).any() and not iso.any():
            fmt = "Mes/día/año con hora" if with_time else "Mes/día/año"
        else:
            continue
        selected.append(col)
        formats.append(fmt)
    if not formats or len(set(formats)) != 1:
        return [], "Día/mes/año"
    return selected, formats[0]


def profile(df: pd.DataFrame) -> pd.DataFrame:
    output = quality(df)
    for field in ["minimo", "mediana", "media", "maximo"]:
        output[field] = pd.Series(dtype="object")
    for i, col in enumerate(df):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            s = df[col]
            for field, val in zip(
                ["minimo", "mediana", "media", "maximo"],
                [s.min(), s.median(), s.mean(), s.max()],
                strict=True,
            ):
                output.at[i, field] = val
    return output


def quality_indicators(df: pd.DataFrame) -> dict:
    if df.empty or not df.size:
        return {"completitud": 0.0, "unicidad": 0.0, "puntaje": 0.0}
    complete = float(1 - df.isna().sum().sum() / df.size)
    unique = float(1 - df.duplicated().mean())
    return {
        "completitud": complete,
        "unicidad": unique,
        "puntaje": 100 * (0.7 * complete + 0.3 * unique),
    }


def apply_filters(
    df: pd.DataFrame, *, text="", categories=None, ranges=None, dates=None, keep_missing=True
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if text:
        mask &= (
            df.astype("string")
            .apply(lambda s: s.str.contains(text, regex=False, case=False, na=False))
            .any(axis=1)
        )
    for col, selected in (categories or {}).items():
        current = df[col].astype("string").isin(selected)
        if keep_missing:
            current |= df[col].isna()
        mask &= current
    for col, (low, high) in {**(ranges or {}), **(dates or {})}.items():
        current = df[col].between(low, high)
        if keep_missing:
            current |= df[col].isna()
        mask &= current.fillna(False)
    return df.loc[mask].copy()


def category_totals(df: pd.DataFrame, dimension: str, metric: str | None = None) -> pd.DataFrame:
    # Una categoría ausente no se mezcla con un nombre textual '(Sin dato)'.
    work = df.dropna(subset=[dimension])
    if metric:
        result = (
            work.groupby(dimension, observed=True)[metric]
            .sum(min_count=1)
            .dropna()
            .reset_index(name="valor")
        )
    else:
        result = work.groupby(dimension, observed=True).size().reset_index(name="valor")
    result[dimension] = result[dimension].astype(str)
    return result.sort_values("valor", ascending=False).reset_index(drop=True)


def pareto_table(df: pd.DataFrame, dimension: str, metric: str | None = None) -> pd.DataFrame:
    if metric and (df[metric].dropna() < 0).any():
        raise ValueError(
            "Pareto requiere valores no negativos; selecciona otra métrica o usa conteo."
        )
    totals = category_totals(df, dimension, metric)
    if totals.empty or totals["valor"].sum() <= 0:
        return pd.DataFrame(columns=[dimension, "valor", "acumulado_pct"])
    totals["acumulado_pct"] = totals["valor"].cumsum() / totals["valor"].sum() * 100
    return totals


def pareto_figure(table: pd.DataFrame, dimension: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=table[dimension], y=table["valor"], name="Aporte"), secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=table[dimension], y=table["acumulado_pct"], name="Acumulado (%)", mode="lines+markers"
        ),
        secondary_y=True,
    )
    fig.add_hline(y=80, line_dash="dot", secondary_y=True)
    fig.update_yaxes(title_text="Aporte", secondary_y=False)
    fig.update_yaxes(title_text="Acumulado (%)", range=[0, 105], secondary_y=True)
    return fig


def time_totals(
    df: pd.DataFrame, date_col: str, metric: str | None, frequency="MS", aggregation="Suma"
) -> pd.DataFrame:
    work = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if work.empty:
        return pd.DataFrame(columns=[date_col, "valor"])
    grouped = work.resample(frequency)
    if metric is None:
        values = grouped.size().astype(float)
        values = values.where(grouped.size() > 0)
    elif aggregation == "Promedio":
        values = grouped[metric].mean()
    else:
        values = grouped[metric].sum(min_count=1)
    # Los meses sin observaciones quedan vacíos, no se inventan ceros de ventas.
    return values.rename("valor").reset_index()


def generate_findings(df: pd.DataFrame, dimension=None, metric=None) -> list[str]:
    if df.empty:
        return ["No hay registros con los filtros activos."]
    quality_data = quality_indicators(df)
    findings = [
        f"La selección contiene {len(df):,} registros y {df.shape[1]} columnas. Completitud: {quality_data['completitud']:.1%}; unicidad de filas: {quality_data['unicidad']:.1%}."
    ]
    if dimension:
        totals = category_totals(df, dimension, metric)
        if not totals.empty and totals["valor"].sum() > 0 and (totals["valor"] >= 0).all():
            top = totals.iloc[0]
            share = top["valor"] / totals["valor"].sum()
            findings.append(
                f"{top[dimension]} reúne el {share:.1%} del {'aporte de ' + metric if metric else 'conteo'} por {dimension}. Revisa esta concentración antes de comparar segmentos."
            )
            count80 = int((totals["valor"].cumsum() / totals["valor"].sum() < 0.8).sum() + 1)
            findings.append(
                f"Se necesitan {count80} de {len(totals)} categorías para alcanzar el 80% del total. Se excluyen categorías vacías."
            )
    missing = df.isna().mean()
    if (missing > 0).any():
        findings.append(
            f"{missing.idxmax()} conserva {missing.max():.1%} de valores faltantes. Confirma su cobertura antes de usarla para decidir."
        )
    numeric = df.select_dtypes("number").columns.tolist()
    pairs = []
    if metric in numeric:
        for other in numeric:
            if other == metric or other.endswith("_atipico"):
                continue
            valid = df[[metric, other]].dropna()
            if len(valid) >= 5 and valid[metric].nunique() > 1 and valid[other].nunique() > 1:
                r = valid[metric].corr(valid[other])
                if pd.notna(r) and abs(r) >= 0.65:
                    pairs.append((abs(r), other, r, len(valid)))
        if pairs:
            _, other, r, count = max(pairs)
            findings.append(
                f"{metric} y {other}: correlación de Pearson {r:.2f} en {count} pares válidos. Puede incluir relaciones derivadas por fórmula; no demuestra causalidad."
            )
    return findings


def build_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    """Libro con pestañas auditables; las cadenas nunca se ejecutan como fórmulas."""
    total_cells = sum((len(frame) + 1) * len(frame.columns) for frame in frames.values())
    if total_cells > MAX_EXCEL_CELLS:
        raise ValueError(
            "El libro supera el límite de 1.000.000 de celdas de esta aplicación. Utiliza CSV o Parquet para los datos completos."
        )
    for frame in frames.values():
        if len(frame) > 1_048_575 or len(frame.columns) > 16_384:
            raise ValueError(
                "El dataset supera los límites de una hoja Excel; utiliza CSV o Parquet."
            )
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Font, PatternFill

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in frames.items():
            safe = frame.copy()
            for col in safe:
                if (
                    pd.api.types.is_datetime64_any_dtype(safe[col])
                    and getattr(safe[col].dt, "tz", None) is not None
                ):
                    safe[col] = safe[col].dt.tz_convert("UTC").dt.tz_localize(None)
                if pd.api.types.is_object_dtype(safe[col]) or pd.api.types.is_string_dtype(
                    safe[col]
                ):
                    safe[col] = safe[col].map(
                        lambda v: ILLEGAL_CHARACTERS_RE.sub("", str(v)) if pd.notna(v) else None
                    )
            safe.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for row in ws:
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="1E40AF")
                cell.font = Font(color="FFFFFF", bold=True)
            for column in ws.columns:
                width = min(
                    45,
                    max(
                        14,
                        max((len(str(c.value or "")) for c in list(column)[:101]), default=14) + 2,
                    ),
                )
                ws.column_dimensions[column[0].column_letter].width = width
    return buffer.getvalue()


def analysis_snapshot(df: pd.DataFrame, specification: dict) -> tuple[list[str], list, dict]:
    """Reconstruye un informe sin ejecutar los widgets de una sección oculta."""
    numeric = df.select_dtypes("number").columns.tolist()
    cats = [
        c
        for c in df
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    default_metric = next(
        (
            c
            for c in numeric
            if any(w in c.lower() for w in ["recaudacion_usd", "ingreso", "revenue", "importe"])
        ),
        numeric[0] if numeric else None,
    )
    dimension = specification.get(
        "dimension", "artista" if "artista" in cats else (cats[0] if cats else None)
    )
    metric = specification.get("metrica", default_metric)
    if dimension not in cats:
        dimension = None
    if metric not in numeric:
        metric = None
    spec = {**specification, "dimension": dimension, "metrica": metric}
    findings, figures = generate_findings(df, dimension, metric), []
    if df.empty:
        return findings, figures, spec
    if dimension:
        grouped = category_totals(df, dimension, metric)
        figures.append(
            (
                "Aporte por categoría",
                px.bar(grouped.head(20), x=dimension, y="valor", template="plotly_dark"),
            )
        )
        try:
            table = pareto_table(df, dimension, metric)
            if not table.empty:
                figures.append(("Pareto", pareto_figure(table.head(30), dimension)))
        except ValueError:
            pass
    date_col = spec.get("fecha")
    if date_col in df and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        table = time_totals(
            df, date_col, metric, spec.get("frecuencia", "MS"), spec.get("agregacion", "Suma")
        )
        figures.append(
            ("Evolución temporal", px.line(table, x=date_col, y="valor", template="plotly_dark"))
        )
    return findings, figures, spec


def build_html_report(
    filename: str,
    markdown: str,
    df: pd.DataFrame,
    audit: pd.DataFrame,
    findings: list[str],
    figures=(),
) -> str:
    blocks = []
    for i, (title, figure) in enumerate(figures):
        # Tema propio del informe exportado, independiente del modo del navegador.
        fig = go.Figure(figure)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#141F33",
            plot_bgcolor="#141F33",
            font_color="#E2E8F0",
        )
        snippet = fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"responsive": True})
        blocks.append(f"<section><h2>{html.escape(title)}</h2>{snippet}</section>")
    audit_note = (
        f"Se muestran {MAX_HTML_AUDIT_ROWS:,} de {len(audit):,} registros. Descarga el CSV de auditoría para obtener el registro completo."
        if len(audit) > MAX_HTML_AUDIT_ROWS
        else "Se muestran todos los registros de la auditoría seleccionada."
    )
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Informe DataClara</title>
<style>body{{background:#0B1120;color:#E2E8F0;font:16px/1.6 system-ui;margin:0;padding:24px}}main{{max-width:1200px;margin:auto}}section{{background:#141F33;padding:24px;margin:20px 0;border-radius:16px;overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border:1px solid #475569}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media print{{body,section{{background:white;color:black}}section{{break-inside:avoid}}}}</style>
</head><body><main><h1>DataClara · Informe de análisis</h1><p>{html.escape(filename)}</p>
<section><h2>Configuración y metodología</h2><pre>{html.escape(markdown)}</pre></section>
<section><h2>Hallazgos de la vista filtrada</h2><ul>{"".join("<li>" + html.escape(f) + "</li>" for f in findings)}</ul></section>
{"".join(blocks)}<section><h2>Perfil de la vista filtrada</h2>{profile(df).to_html(index=False, escape=True)}</section>
<section><h2>Muestra de la vista filtrada (hasta 100 filas)</h2>{df.head(100).to_html(index=False, escape=True)}</section>
<section><h2>Registro de limpieza</h2><p>{audit_note}</p>{audit.head(MAX_HTML_AUDIT_ROWS).to_html(index=False, escape=True)}</section>
<footer>Datos procesados localmente. Las asociaciones no demuestran causalidad. Para PDF, usa Imprimir → Guardar como PDF en el navegador.</footer>
</main></body></html>"""
