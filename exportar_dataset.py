"""Exporta una versión limpia y normalizada del dataset incluido."""

from pathlib import Path

from analysis import Options, clean, load_csv, normalize


def main():
    root = Path(__file__).resolve().parent
    original, _ = load_csv((root / "my_file (1).csv").read_bytes())
    cleaned, audit, _ = clean(original, Options())
    columns = [
        "recaudacion_usd",
        "recaudacion_ajustada_usd",
        "conciertos",
        "promedio_reportado_usd",
        "ingreso_por_concierto_usd",
    ]
    normalized, parameters = normalize(cleaned, columns)
    destination = root / "resultados"
    destination.mkdir(exist_ok=True)
    for name, frame in [
        ("datos_limpios.csv", cleaned),
        ("datos_normalizados.csv", normalized),
        ("parametros_normalizacion.csv", parameters),
        ("registro_limpieza.csv", audit),
    ]:
        frame.to_csv(destination / name, index=False, encoding="utf-8-sig")
    print(f"Exportados {len(normalized)} registros a {destination}")


if __name__ == "__main__":
    main()
