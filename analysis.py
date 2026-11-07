"""Lectura, limpieza resumida y normalización de datasets."""

import csv
import io
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

NULL_TOKENS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nan",
    "-",
    "--",
    "?",
    "sin dato",
    "sin datos",
    "no disponible",
    "n/d",
    "s/d",
}
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


def load_csv(
    data: bytes, separator: str = "Automático", encoding: str = "Automático"
) -> tuple[pd.DataFrame, str]:
    if not data or len(data) > 50 * 1024 * 1024:
        raise ValueError("El archivo debe contener datos y pesar menos de 50 MB.")
    text = None
    used_encoding = encoding
    for candidate in (
        ["utf-8-sig", "cp1252", "latin-1"] if encoding == "Automático" else [encoding]
    ):
        try:
            text = data.decode(candidate)
            used_encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(
            "No se pudo decodificar el archivo. Selecciona otra codificación."
        )
    sep = separator
    if sep == "Automático":
        try:
            sep = csv.Sniffer().sniff(text[:65536], delimiters=";,\t|").delimiter
        except csv.Error:
            sep = ","
    # Validar encabezados antes de que pandas renombre silenciosamente los repetidos.
    header = next(csv.reader(io.StringIO(text), delimiter=sep), [])
    normalized = [unicodedata.normalize("NFKC", str(x)).strip().lower() for x in header]
    if (
        not normalized
        or any(not x for x in normalized)
        or len(set(normalized)) != len(normalized)
    ):
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
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        pd.errors.ParserWarning,
    ) as exc:
        raise ValueError(
            f"CSV inválido. Revisa el separador y las filas: {exc}"
        ) from exc
    if df.empty:
        raise ValueError("El archivo tiene encabezados pero no contiene registros.")
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df, f"{used_encoding} · separador {sep!r}"


def number(value, decimal=".") -> float:
    """Convierte solo formatos inequívocos según la convención seleccionada."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return float(value) if np.isfinite(value) else np.nan
    text = re.sub(r"\[[^\]]*\]", "", str(value))
    text = re.sub(r"[\s$€£¥]", "", text)
    text = re.sub(r"(?i)(USD|COP|EUR|MXN|ARS|CLP|PEN)", "", text)
    text = text.removesuffix("%")  # Porcentajes en puntos porcentuales: 45% -> 45.
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    thousands = "," if decimal == "." else "."
    d, t = re.escape(decimal), re.escape(thousands)
    pattern = rf"[+-]?(?:\d+|\d{{1,3}}(?:{t}\d{{3}})+)(?:{d}\d+)?"
    if not re.fullmatch(pattern, text):
        return np.nan
    result = float(text.replace(thousands, "").replace(decimal, "."))
    return result if np.isfinite(result) else np.nan


def parse_dates(series: pd.Series, date_format: str) -> pd.Series:
    """El modo mixto es explícito y preserva el orden ISO año-mes-día."""
    if date_format != "mixed_dayfirst":
        return pd.to_datetime(
            series, format=date_format, errors="coerce", utc=True
        ).dt.tz_localize(None)
    iso = series.astype("string").str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    output = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    output.loc[iso] = pd.to_datetime(
        series.loc[iso], format="ISO8601", errors="coerce", utc=True
    )
    output.loc[~iso] = pd.to_datetime(
        series.loc[~iso], format="mixed", dayfirst=True, errors="coerce", utc=True
    )
    return output.dt.tz_localize(None)


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
                "faltantes_pct": round(float(s.isna().mean() * 100), 2)
                if len(s)
                else 0,
                "unicos": int(s.nunique()),
                "atipicos_IQR": outliers,
            }
        )
    return pd.DataFrame(rows)


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
        low, high, mean, std = (
            values.min(),
            values.max(),
            values.mean(),
            values.std(ddof=0),
        )
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


def load_dataset(data, filename, separator="Automático", sheet=0):
    if not data or len(data) > 50 * 1024 * 1024:
        raise ValueError("El archivo debe contener datos y pesar hasta 50 MB.")
    if filename.lower().endswith(".csv"):
        return load_csv(data, separator)[0]
    if not filename.lower().endswith(".xlsx"):
        raise ValueError("Usa un archivo CSV o Excel (.xlsx).")
    table = pd.read_excel(
        io.BytesIO(data),
        sheet_name=sheet,
        header=None,
        dtype=object,
        keep_default_na=False,
    )
    if len(table) < 2:
        raise ValueError("La hoja debe contener encabezados y registros.")
    headers = [str(value).strip() for value in table.iloc[0]]
    keys = [unicodedata.normalize("NFKC", value).casefold() for value in headers]
    if any(not value for value in keys) or len(set(keys)) != len(keys):
        raise ValueError("Los encabezados deben ser únicos y no estar vacíos.")
    result = table.iloc[1:].copy()
    result.columns = headers
    return result.reset_index(drop=True)


def suggest_numeric(df, decimal="."):
    """Sugiere números con una muestra; protege identificadores y ceros iniciales."""
    selected = []
    for col in df:
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", col).lower()
        if set(re.split(r"[^a-z]+", name)) & {
            "id",
            "code",
            "codigo",
            "invoice",
            "no",
            "sku",
            "telefono",
            "postal",
        }:
            continue
        sample = df[col].dropna().astype(str).head(500)
        if (
            len(sample)
            and not sample.str.fullmatch(r"0\d+").any()
            and sample.map(lambda v: pd.notna(number(v, decimal))).mean() >= 0.9
        ):
            selected.append(col)
    return selected


def clean(df, options):
    """Aplica reglas sobre una copia y cuenta cambios por operación, sin auditar celda a celda."""
    result, audit = df.copy(deep=True), []

    def record(col, operation, count):
        if count:
            audit.append(
                {"columna": col, "operacion": operation, "cantidad": int(count)}
            )

    for col in result.select_dtypes(include=["object", "string"]):
        before = result[col].astype("string")
        after = (
            before.str.normalize("NFKC")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        after = after.mask(after.str.lower().isin(NULL_TOKENS))
        if options["lowercase"] and col not in options["numeric"] + options["dates"]:
            after = after.str.lower()
        changed = before.fillna("").ne(after.fillna(""))
        record(col, "Espacios, texto y valores vacíos", changed.sum())
        result[col] = after
    for col in options["numeric"]:
        before = result[col]
        lookup = {
            value: number(value, options["decimal"])
            for value in before.dropna().unique()
        }
        result[col] = before.map(lookup).astype(float)
        record(col, "Conversión numérica", before.notna().sum())
        record(
            col,
            "Número inválido convertido a vacío",
            (before.notna() & result[col].isna()).sum(),
        )
    for col in options["dates"]:
        before = result[col]
        result[col] = parse_dates(before, options["date_format"])
        record(col, "Conversión a fecha", before.notna().sum())
        record(
            col,
            "Fecha inválida convertida a vacío",
            (before.notna() & result[col].isna()).sum(),
        )
    empty = result.isna().all(axis=1)
    record("(fila)", "Eliminar fila vacía", empty.sum())
    result = result.loc[~empty].copy()
    if options["deduplicate"]:
        duplicate = result.duplicated()
        record("(fila)", "Eliminar duplicado", duplicate.sum())
        result = result.loc[~duplicate].copy()
    # El umbral se aplica antes de rellenar faltantes.
    for col in list(result):
        if len(result) and result[col].isna().mean() >= options["null_threshold"]:
            record(col, "Eliminar columna por proporción de nulos", 1)
            result = result.drop(columns=col)
    if options["missing"] == "Eliminar filas":
        missing = result.isna().any(axis=1)
        record("(fila)", "Eliminar fila incompleta", missing.sum())
        result = result.loc[~missing].copy()
    elif options["missing"] == "Mediana / Sin dato":
        for col in result:
            s = result[col]
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
                continue
            value = s.median() if pd.api.types.is_numeric_dtype(s) else "Sin dato"
            if pd.notna(value):
                record(col, "Rellenar faltantes", s.isna().sum())
                result[col] = s.fillna(value)
    if options["outliers"] == "Recortar (IQR)":
        for col in options["numeric"]:
            if col not in result or result[col].notna().sum() < 4:
                continue
            q1, q3 = result[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                clipped = result[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                record(
                    col,
                    "Recortar atípicos IQR",
                    (result[col].notna() & result[col].ne(clipped)).sum(),
                )
                result[col] = clipped
    return result.reset_index(drop=True), pd.DataFrame(
        audit, columns=["columna", "operacion", "cantidad"]
    )
