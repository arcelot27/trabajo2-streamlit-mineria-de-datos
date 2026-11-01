"""Carga, limpieza trazable y análisis. No modifica el archivo de origen."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TOUR_COLUMNS = {
    "rank": "posicion",
    "peak": "mejor_posicion",
    "all time peak": "mejor_posicion_historica",
    "actual gross": "recaudacion_usd",
    "adjusted gross (in 2022 dollars)": "recaudacion_ajustada_usd",
    "artist": "artista",
    "tour title": "gira",
    "year(s)": "periodo",
    "shows": "conciertos",
    "average gross": "promedio_reportado_usd",
    "ref.": "referencias",
}
MONEY = ["recaudacion_usd", "recaudacion_ajustada_usd", "promedio_reportado_usd"]
COUNTS = ["posicion", "mejor_posicion", "mejor_posicion_historica", "conciertos"]
AUDIT_COLUMNS = ["fila_csv", "columna", "antes", "despues", "motivo", "cantidad"]


@dataclass
class Options:
    deduplicate: bool = True
    trim: bool = True
    missing: str = "Conservar"
    numeric_columns: list[str] = field(default_factory=list)
    decimal: str = "."
    normalize_names: bool = False
    text_columns: list[str] = field(default_factory=list)
    case_mode: str = "Conservar"
    remove_accents: bool = False
    date_columns: list[str] = field(default_factory=list)
    date_format: str = "%d/%m/%Y"
    duplicate_subset: list[str] = field(default_factory=list)
    drop_empty_rows: bool = False
    drop_empty_columns: bool = False
    drop_constant_columns: bool = False
    null_threshold: float | None = None
    missing_columns: list[str] = field(default_factory=list)
    outlier_columns: list[str] = field(default_factory=list)
    outlier_method: str = "IQR"
    outlier_factor: float = 1.5
    outlier_action: str = "Conservar"
    audit_mode: str = "Automático"


NULL_TOKENS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nan",
    "n.a.",
    "nil",
    "-",
    "--",
    "---",
    "?",
    "s/d",
    "sin dato",
    "sin datos",
    "sin información",
    "sin informacion",
    "no aplica",
    "no disponible",
    "#n/a",
    "undefined",
    "n/d",
}


def column_names(columns) -> dict[str, str]:
    """Nombres snake_case únicos, con sufijos para evitar colisiones."""
    used, mapping = set(), {}
    for col in columns:
        name = unicodedata.normalize("NFKD", str(col))
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "columna"
        if base[0].isdigit():
            base = "col_" + base
        name, suffix = base, 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        mapping[col] = name
        used.add(name)
    return mapping


def parse_dates(series: pd.Series, date_format: str) -> pd.Series:
    """El modo mixto es explícito y preserva el orden ISO año-mes-día."""
    if date_format != "mixed_dayfirst":
        return pd.to_datetime(series, format=date_format, errors="coerce", utc=True).dt.tz_localize(
            None
        )
    iso = series.astype("string").str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    output = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    output.loc[iso] = pd.to_datetime(series.loc[iso], format="ISO8601", errors="coerce", utc=True)
    output.loc[~iso] = pd.to_datetime(
        series.loc[~iso], format="mixed", dayfirst=True, errors="coerce", utc=True
    )
    return output.dt.tz_localize(None)


def load_csv(
    data: bytes, separator: str = "Automático", encoding: str = "Automático"
) -> tuple[pd.DataFrame, str]:
    if not data or len(data) > 50 * 1024 * 1024:
        raise ValueError("El archivo debe contener datos y pesar menos de 50 MB.")
    text = None
    used_encoding = encoding
    for candidate in ["utf-8-sig", "cp1252", "latin-1"] if encoding == "Automático" else [encoding]:
        try:
            text = data.decode(candidate)
            used_encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("No se pudo decodificar el archivo. Selecciona otra codificación.")
    sep = separator
    if sep == "Automático":
        try:
            sep = csv.Sniffer().sniff(text[:65536], delimiters=";,\t|").delimiter
        except csv.Error:
            sep = ","
    # Validar encabezados antes de que pandas renombre silenciosamente los repetidos.
    header = next(csv.reader(io.StringIO(text), delimiter=sep), [])
    normalized = [unicodedata.normalize("NFKC", str(x)).strip().lower() for x in header]
    if not normalized or any(not x for x in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("Los encabezados deben ser únicos y no estar vacíos.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                dtype="string",
                keep_default_na=False,
                index_col=False,
                on_bad_lines="error",
            )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, pd.errors.ParserWarning) as exc:
        raise ValueError(f"CSV inválido. Revisa el separador y las filas: {exc}") from exc
    if df.empty:
        raise ValueError("El archivo tiene encabezados pero no contiene registros.")
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df, f"{used_encoding} · separador {repr(sep)}"


def key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().lower()


def is_tours(df: pd.DataFrame) -> bool:
    return {"artist", "tour title", "actual gross", "shows", "year(s)"}.issubset(
        {key(c) for c in df.columns}
    )


def number(value, decimal=".") -> float:
    """Convierte solo formatos inequívocos según la convención seleccionada."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return float(value) if np.isfinite(value) else np.nan
    text = re.sub(r"\[[^\]]*\]", "", str(value))
    text = re.sub(r"[\s$€£¥]", "", text)
    text = re.sub(r"(?i)(USD|COP|EUR|MXN|ARS|CLP|PEN)", "", text)
    if text.endswith("%"):
        text = text[:-1]  # Porcentajes en puntos porcentuales: 45% -> 45.
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    thousands = "," if decimal == "." else "."
    d, t = re.escape(decimal), re.escape(thousands)
    pattern = rf"[+-]?(?:\d+|\d{{1,3}}(?:{t}\d{{3}})+)(?:{d}\d+)?"
    if not re.fullmatch(pattern, text):
        return np.nan
    result = float(text.replace(thousands, "").replace(decimal, "."))
    return result if np.isfinite(result) else np.nan


def quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df:
        s = df[col]
        outliers = 0
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append(
            {
                "columna": col,
                "tipo": str(s.dtype),
                "faltantes": int(s.isna().sum()),
                "faltantes_pct": round(float(s.isna().mean() * 100), 2) if len(s) else 0,
                "unicos": int(s.nunique()),
                "atipicos_IQR": outliers,
            }
        )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, options: Options) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    result = df.copy().reset_index(drop=True)
    audit = []
    summary_mode = options.audit_mode == "Resumido" or (
        options.audit_mode == "Automático" and len(result) > 50_000
    )
    if options.audit_mode not in {"Automático", "Resumido", "Detalle completo"}:
        raise ValueError("Modo de auditoría desconocido.")

    def record(idx, col, before, after, reason):
        audit.append(
            {
                "fila_csv": int(idx) + 2 if idx is not None else pd.NA,
                "columna": col,
                "antes": "" if pd.isna(before) else str(before),
                "despues": "" if pd.isna(after) else str(after),
                "motivo": reason,
                "cantidad": 1,
            }
        )

    def record_batch(indices, col, before, after, reason):
        """Resume millones de conversiones sin construir millones de diccionarios."""
        count = len(indices)
        if not count:
            return
        if summary_mode:
            first = indices[0]
            old = before.loc[first] if isinstance(before, pd.Series) else before
            new = after.loc[first] if isinstance(after, pd.Series) else after
            record(None, col, old, new, reason)
            audit[-1]["cantidad"] = count
            audit[-1]["motivo"] += (
                f" · resumen de {count:,} cambios; antes/después muestran el primer ejemplo"
            )
        else:
            old_values = before.loc[indices] if isinstance(before, pd.Series) else [before] * count
            new_values = after.loc[indices] if isinstance(after, pd.Series) else [after] * count
            for idx, old, new in zip(indices, old_values, new_values, strict=True):
                record(idx, col, old, new, reason)

    for col in result:
        if not (
            pd.api.types.is_string_dtype(result[col]) or pd.api.types.is_object_dtype(result[col])
        ):
            continue
        before = result[col].astype("string")
        new = before.copy()
        if options.trim:
            new = new.str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()
        if col in options.text_columns:
            if options.remove_accents:
                new = new.str.normalize("NFKD").map(
                    lambda value: "".join(c for c in value if not unicodedata.combining(c)),
                    na_action="ignore",
                )
            if options.case_mode == "Minúsculas":
                new = new.str.lower()
            elif options.case_mode == "Mayúsculas":
                new = new.str.upper()
            elif options.case_mode == "Título":
                new = new.str.title()
        new = new.mask(new.str.strip().str.lower().isin(NULL_TOKENS))
        changed = before.notna() & (new.isna() | before.ne(new).fillna(False))
        record_batch(
            before.index[changed], col, before, new, "Normalización de texto / valor vacío"
        )
        result[col] = new

    tours = is_tours(result)
    if tours:
        result = result.rename(columns={c: TOUR_COLUMNS.get(key(c), c) for c in result})
        if result.columns.duplicated().any():
            raise ValueError(
                "Hay columnas que entran en conflicto con los nombres del esquema de giras."
            )
    numeric = [c for c in MONEY + COUNTS if c in result] if tours else options.numeric_columns
    for col in numeric:
        if col not in result:
            continue
        before = result[col]
        # Los precios y cantidades repetidos se interpretan una sola vez.
        lookup = {
            value: number(value, "." if tours else options.decimal)
            for value in before.dropna().unique()
        }
        converted = before.map(lookup).astype("float64")
        if tours:
            invalid = converted.lt(0)
            if col in COUNTS:
                invalid |= converted.mod(1).ne(0)
            if col == "conciertos":
                invalid |= converted.eq(0)
            converted = converted.mask(invalid)
        present = before.notna()
        record_batch(
            before.index[present & converted.notna()], col, before, converted, "Conversión numérica"
        )
        record_batch(
            before.index[present & converted.isna()],
            col,
            before,
            converted,
            "Formato o valor inválido; queda vacío",
        )
        result[col] = converted

    if set(options.date_columns).intersection(numeric):
        raise ValueError("Una columna no puede ser numérica y fecha al mismo tiempo.")
    if not tours:
        for col in options.date_columns:
            if col not in result:
                continue
            converted = parse_dates(result[col], options.date_format)
            present = result[col].notna()
            record_batch(
                result.index[present & converted.notna()],
                col,
                result[col],
                converted,
                "Conversión a fecha (UTC para zonas explícitas)",
            )
            record_batch(
                result.index[present & converted.isna()],
                col,
                result[col],
                converted,
                "Fecha inválida; queda vacía",
            )
            result[col] = converted

    if options.drop_empty_rows:
        empty = result.isna().all(axis=1)
        record_batch(
            result.index[empty], "(fila)", "Vacía", "Eliminada", "Fila completamente vacía"
        )
        result = result.loc[~empty].copy()

    if options.outlier_action != "Conservar":
        if options.outlier_factor <= 0:
            raise ValueError("El factor de atípicos debe ser positivo.")
        if tours and options.outlier_action != "Marcar":
            raise ValueError("En giras los atípicos solo se marcan para conservar sus indicadores.")
        masks, bounds = {}, {}
        # Límites calculados una vez; eliminar no depende del orden de las columnas.
        for col in options.outlier_columns:
            if (
                col not in result
                or not pd.api.types.is_numeric_dtype(result[col])
                or pd.api.types.is_bool_dtype(result[col])
            ):
                continue
            s = result[col]
            if s.notna().sum() < 4:
                continue
            if options.outlier_method == "IQR":
                q1, q3 = s.quantile([0.25, 0.75])
                scale, center = q3 - q1, None
                low, high = q1 - options.outlier_factor * scale, q3 + options.outlier_factor * scale
            elif options.outlier_method == "Z-score":
                scale, center = s.std(ddof=0), s.mean()
                low, high = (
                    center - options.outlier_factor * scale,
                    center + options.outlier_factor * scale,
                )
            else:
                raise ValueError("Método de atípicos desconocido.")
            if pd.isna(scale) or scale == 0:
                continue
            masks[col] = ((s < low) | (s > high)).fillna(False)
            bounds[col] = (low, high)
        for col, mask in masks.items():
            if options.outlier_action == "Marcar":
                flag = f"{col}_atipico"
                if flag in result:
                    raise ValueError(
                        f"Ya existe la columna {flag}; cambia la selección de atípicos."
                    )
                result[flag] = mask.astype(bool)
                record_batch(
                    result.index[mask],
                    flag,
                    pd.NA,
                    True,
                    f"Atípico marcado ({options.outlier_method}, factor {options.outlier_factor})",
                )
            elif options.outlier_action == "Recortar":
                clipped = result[col].clip(*bounds[col])
                record_batch(
                    result.index[mask],
                    col,
                    result[col],
                    clipped,
                    f"Recorte de atípico ({options.outlier_method})",
                )
                result[col] = clipped
        if options.outlier_action == "Eliminar" and masks:
            remove = pd.DataFrame(masks).any(axis=1)
            record_batch(
                result.index[remove],
                "(fila)",
                "Atípica",
                "Eliminada",
                "Atípico en columnas seleccionadas; límites fijos",
            )
            result = result.loc[~remove].copy()

    if tours:
        if "gira" in result:
            for idx, value in result["gira"].items():
                if pd.isna(value):
                    continue
                new = re.sub(r"\[[^\]]*\]|[†‡*]", "", str(value)).strip()
                if new != value:
                    record(
                        idx,
                        "gira",
                        value,
                        new,
                        "Retiro de marcas bibliográficas; ver archivo original",
                    )
                result.at[idx, "gira"] = new or pd.NA
        for name in ["anio_inicio", "anio_fin"]:
            result[name] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        for idx, value in result["periodo"].items():
            if pd.isna(value):
                continue
            match = re.fullmatch(r"(\d{4})(?:\s*[–—-]\s*(\d{4}))?", str(value))
            if match and int(match[1]) <= int(match[2] or match[1]):
                result.at[idx, "anio_inicio"] = int(match[1])
                result.at[idx, "anio_fin"] = int(match[2] or match[1])
            else:
                record(idx, "periodo", value, pd.NA, "Periodo inválido; años derivados vacíos")
        result["ingreso_por_concierto_usd"] = result["recaudacion_usd"] / result["conciertos"]
        if "promedio_reportado_usd" in result:
            calculated = result["ingreso_por_concierto_usd"]
            reported = result["promedio_reportado_usd"]
            result["promedio_discrepante"] = (
                reported.notna() & calculated.notna() & ((reported - calculated).abs() > 1)
            )
        # Las filas idénticas se detectan después de normalizar y convertir.
    if options.deduplicate:
        subset = [TOUR_COLUMNS.get(key(c), c) if tours else c for c in options.duplicate_subset]
        duplicates = result.duplicated(subset=subset or None)
        record_batch(
            result.index[duplicates],
            "(fila)",
            "Duplicada",
            "Eliminada",
            "Duplicado por clave seleccionada" if subset else "Duplicado exacto tras normalizar",
        )
        result = result.loc[~duplicates].copy()
    # Quitar columnas vacías antes de imputar evita fabricar una columna constante.
    if not tours and options.drop_empty_columns:
        for col in list(result.columns):
            if result[col].isna().all():
                record(None, col, "Columna", "Eliminada", "Columna vacía")
                result = result.drop(columns=[col])
    # No imputar dinero ni rankings del esquema especializado: fabricarían indicadores.
    if not tours and options.missing in {
        "Mediana / Sin dato",
        "Media / Moda",
        "Cero / Sin dato",
        "Moda",
        "Valor anterior",
    }:
        for col in options.missing_columns or list(result.columns):
            if col not in result:
                continue
            series = result[col]
            if pd.api.types.is_datetime64_any_dtype(series) and options.missing != "Valor anterior":
                continue
            numeric_col = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
                series
            )
            mode = series.mode(dropna=True)
            if options.missing == "Valor anterior":
                filled = series.ffill()
                record_batch(
                    result.index[series.isna() & filled.notna()],
                    col,
                    pd.NA,
                    filled,
                    "Imputación con valor anterior (orden original)",
                )
                result[col] = filled
                continue
            if pd.api.types.is_bool_dtype(series) and options.missing not in {
                "Media / Moda",
                "Moda",
            }:
                continue
            if options.missing == "Mediana / Sin dato":
                fill = series.median() if numeric_col else "Sin dato"
            elif options.missing == "Media / Moda":
                fill = series.mean() if numeric_col else (mode.iloc[0] if not mode.empty else pd.NA)
            elif options.missing == "Cero / Sin dato":
                fill = 0 if numeric_col else "Sin dato"
            else:
                fill = mode.iloc[0] if not mode.empty else pd.NA
            if pd.isna(fill):
                continue
            record_batch(
                result.index[result[col].isna()],
                col,
                pd.NA,
                fill,
                "Imputación seleccionada por el usuario",
            )
            result[col] = result[col].fillna(fill)
    elif options.missing == "Eliminar filas incompletas":
        # Solo columnas de entrada; campos derivados no deciden la eliminación.
        subset = [TOUR_COLUMNS.get(key(c), c) if tours else c for c in df.columns]
        subset = [
            TOUR_COLUMNS.get(key(c), c) if tours else c for c in options.missing_columns
        ] or subset
        subset = [c for c in subset if c in result]
        invalid = result[subset].isna().any(axis=1)
        record_batch(
            result.index[invalid],
            "(fila)",
            "Incompleta",
            "Eliminada",
            "Política de faltantes seleccionada",
        )
        result = result.loc[~invalid].copy()
    if not tours:
        drop = []
        for col in result:
            reason = None
            if options.drop_empty_columns and result[col].isna().all():
                reason = "Columna vacía"
            elif options.drop_constant_columns and result[col].nunique(dropna=False) <= 1:
                reason = "Columna constante"
            elif (
                options.null_threshold is not None
                and len(result)
                and result[col].isna().mean() >= options.null_threshold
            ):
                reason = "Columna con proporción de nulos igual o superior al umbral"
            if reason:
                record(None, col, "Columna", "Eliminada", reason)
                drop.append(col)
        result = result.drop(columns=drop)
        if options.normalize_names:
            mapping = column_names(result.columns)
            for old, new in mapping.items():
                if old != new:
                    record(None, old, old, new, "Nombre de columna normalizado")
            result = result.rename(columns=mapping)
    return result.reset_index(drop=True), pd.DataFrame(audit, columns=AUDIT_COLUMNS), tours


def artist_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = df.dropna(subset=["artista"])
    grouped = valid.groupby("artista", as_index=False).agg(
        recaudacion=(metric, lambda s: s.sum(min_count=1)),
        conciertos=("conciertos", lambda s: s.sum(min_count=1)),
        giras=("gira", "size"),
    )
    # Media ponderada: excluir pares incompletos del numerador y denominador.
    pairs = valid.dropna(subset=[metric, "conciertos"])
    totals = pairs.groupby("artista")[[metric, "conciertos"]].sum()
    weighted = totals[metric] / totals["conciertos"].replace(0, np.nan)
    grouped["por_concierto"] = grouped["artista"].map(weighted)
    return grouped.sort_values("recaudacion", ascending=False)


NORMALIZATION_COLUMNS = [
    "columna",
    "metodo",
    "validos",
    "faltantes",
    "minimo",
    "maximo",
    "media",
    "desviacion",
    "estado",
]


def normalize(
    df: pd.DataFrame, columns: list[str], method: str = "Min–Max (0 a 1)"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escala columnas seleccionadas sobre el dataset limpio completo, sin imputar.

    Las constantes se convierten a cero y las columnas vacías permanecen vacías.
    La desviación poblacional (ddof=0) coincide con el escalado Z-score habitual.
    """
    if method not in {"Min–Max (0 a 1)", "Z-score"}:
        raise ValueError("Método de normalización desconocido.")
    result = df.copy(deep=True)
    parameters = []
    for col in dict.fromkeys(columns):
        if (
            col not in df
            or not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            raise ValueError(f"La columna {col!r} debe ser numérica y no booleana.")
        values = df[col].astype("float64")
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"La columna {col!r} contiene valores infinitos.")
        low, high, mean, std = values.min(), values.max(), values.mean(), values.std(ddof=0)
        valid = int(values.notna().sum())
        state = "Normalizada"
        if not valid:
            state = "Sin datos válidos; permanece vacía"
            result[col] = values
        elif high == low:
            state = "Constante; valores válidos convertidos a 0"
            result[col] = values.where(values.isna(), 0.0)
        elif method == "Min–Max (0 a 1)":
            result[col] = (values - low) / (high - low)
        else:
            result[col] = (values - mean) / std
        parameters.append(
            {
                "columna": col,
                "metodo": method,
                "validos": valid,
                "faltantes": int(values.isna().sum()),
                "minimo": low,
                "maximo": high,
                "media": mean,
                "desviacion": std,
                "estado": state,
            }
        )
    return result, pd.DataFrame(parameters, columns=NORMALIZATION_COLUMNS)
