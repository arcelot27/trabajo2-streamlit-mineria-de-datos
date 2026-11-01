# DataClara Analytics

Aplicación de Python y Streamlit para cargar datasets, limpiar datos, normalizar variables, generar gráficas y calcular predicciones.

Los resultados se calculan con el archivo que sube el usuario. La aplicación abre sin datos precargados.

## Ejecutar

Requiere Python 3.13. Desde esta carpeta:

```shell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Windows también puedes abrir `iniciar.bat` después de instalar las dependencias.

## Uso

1. Sube un CSV, Excel, JSON o Parquet de hasta 50 MB.
2. Revisa los tipos detectados y ajusta las reglas de limpieza.
3. Consulta calidad, normalización, análisis general y gráficas.
4. Abre **Predicción y decisiones** para proyectar una métrica con fechas.
5. Descarga los datos limpios, informes y resultados desde **Descargas**.

La lectura y limpieza se reutilizan al cambiar de sección. La auditoría se resume automáticamente en archivos de más de 50.000 filas. Las tablas están paginadas y las descargas conservan todas las filas.

Las predicciones muestran errores de evaluación y un rango orientativo; no garantizan resultados futuros. Excel se limita a 1.000.000 de celdas entre todas sus hojas; los archivos mayores se pueden descargar como CSV o Parquet.

## Streamlit Community Cloud

Configuración para el despliegue:

- Rama: `main`.
- Archivo principal: `app.py`.
- Python: `3.13`, seleccionado en **Advanced settings**.
- Dependencias: `requirements.txt`.
- Tema y configuración: `.streamlit/config.toml`.

La conexión con GitHub y la publicación se completan por separado. Los datasets particulares, resultados y secretos están excluidos de Git.

## Archivos de la aplicación

| Archivo | Función |
| --- | --- |
| `app.py` | Interfaz y descargas |
| `analysis.py` | Limpieza y normalización |
| `data_tools.py` | Lectura de archivos, análisis e informes |
| `studio_ui.py` | Controles y filtros |
| `forecasting.py` | Predicción y escenarios |

Las carpetas de datasets y resultados contienen archivos locales del usuario. La muestra de ventas de `sample_data` se utiliza únicamente cuando se elige la demostración.
