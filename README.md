# DataClara Analytics

Aplicación de análisis de datos con Python y Streamlit: carga de archivos, limpieza auditable, normalización, gráficas y proyecciones con evaluación temporal.

La aplicación abre sin datos precargados. Los resultados se calculan a partir del archivo que sube cada usuario y de sus filtros.

## Instalación y ejecución

Usa **Python 3.13**. Las versiones de las dependencias directas están fijadas en `requirements.txt`.

```shell
python -m venv .venv
```

Activa el entorno:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```shell
# Linux y macOS
source .venv/bin/activate
```

Después, desde la raíz del proyecto:

```shell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre la URL local que indica Streamlit. En Windows también puedes usar `iniciar.bat` después de instalar las dependencias.

## Funciones

- **Lectura:** CSV/TSV, Excel `.xlsx`/`.xls`, JSON/JSONL y Parquet. Límite de 50 MB; selección de hoja Excel y opciones de separador/codificación.
- **Limpieza:** espacios, nulos escritos como texto, tipos numéricos y fechas, duplicados por fila o clave, imputación configurable, columnas vacías/constantes y tratamiento explícito de atípicos.
- **Normalización:** Min–Max y Z-score con parámetros descargables. Los indicadores de negocio conservan sus unidades.
- **Exploración:** filtros de texto, categoría, número y fecha; histogramas, dispersión, cajas, correlaciones, Pareto, composición y series temporales.
- **Predicción y decisiones:** último periodo, promedio reciente y tendencia lineal; prueba cronológica, métricas de error, banda orientativa y escenarios configurables.
- **Exportación:** CSV, JSON, Parquet, libro Excel, informe HTML sin conexión y paquete ZIP. El HTML se puede imprimir a PDF desde el navegador.
- **Interfaz:** modo oscuro predeterminado y secciones que se calculan al abrirse.

## Flujo de trabajo

1. Selecciona **Subir un archivo** y cárgalo. Las fuentes locales y la demostración son opciones adicionales.
2. Revisa las sugerencias de tipos y ajusta **Reglas de limpieza** y **Limpieza avanzada**. Los posibles identificadores se conservan como texto; las convenciones de fechas ambiguas requieren revisión.
3. Verifica el diagnóstico y el registro de cambios en **Calidad y limpieza**.
4. Aplica filtros y explora los datos. La calidad y el escalado corresponden al dataset limpio completo; el análisis y las predicciones usan la selección filtrada.
5. En **Descargas**, genera los formatos adicionales o prepara el ZIP con la configuración activa.

Las zonas horarias explícitas se convierten a UTC y se guardan sin zona. Los porcentajes se interpretan como puntos porcentuales: `45%` se convierte en `45`, no en `0,45`.

## Rendimiento y límites

La lectura, limpieza y normalización se reutilizan dentro de la sesión cuando sus entradas no cambian. Al cargar otro archivo se liberan las exportaciones y copias derivadas del anterior.

La auditoría automática resume por columna y operación cuando hay más de 50.000 filas: `cantidad` indica las celdas o filas afectadas, y antes/después muestran un ejemplo. **Detalle completo** registra los cambios individuales y puede consumir más memoria. Ambos modos producen los mismos datos limpios.

Los histogramas cuentan todo el dataset en el servidor. Las tablas están paginadas y las vistas previas son limitadas; las descargas conservan todas las filas. Las gráficas de caja y dispersión usan una muestra reproducible de hasta 5.000 filas e indican esa limitación.

Excel admite como máximo **1.000.000 de celdas entre todas las hojas** en esta aplicación para limitar el uso de memoria. Cuando se supera, continúan disponibles CSV, JSON, Parquet e informe HTML. El HTML muestra hasta 5.000 registros de auditoría; el CSV conserva la auditoría seleccionada completa.

## Interpretación de predicciones

Los modelos usan exclusivamente las fechas y métricas del archivo cargado. La evaluación separa cronológicamente entrenamiento, validación de la banda y prueba. No se elige automáticamente el modelo ganador sobre ese mismo tramo de prueba.

La banda usa el percentil 90 del error absoluto de validación: es orientativa y no garantiza una cobertura del 90%. Los periodos sin datos no se sustituyen por cero salvo elección expresa; por defecto se usa el tramo continuo más reciente. Se requieren al menos 16 periodos regulares.

La proyección comienza después del último periodo utilizado. Si el último periodo del archivo era incompleto y se excluyó, la primera proyección lo cubre. Un dataset antiguo produce fechas proyectadas que también pueden ser históricas respecto al calendario actual.

Los escenarios porcentuales son hipótesis del usuario. Los importes no equivalen a utilidad y las correlaciones no demuestran causalidad. La comparación de productos ayuda a revisar demanda y devoluciones; sin existencias ni costos no se puede obtener un punto de reorden o una utilidad real.

Para el esquema opcional de giras musicales, la recaudación se utiliza tal como figura en el archivo. La atribución al año inicial no equivale a ingresos anuales y el ajuste monetario reportado no se recalcula.

## Procesamiento por consola

```shell
python procesar.py "archivo.csv" --salida resultados_lote
python procesar.py "ventas.xlsx" --hoja Ventas --fechas Fecha --formato-fecha "%d/%m/%Y" --salida resultados_ventas
```

Opciones adicionales: `--numericas`, `--decimal .` o `,`, y `--normalizacion minmax` o `zscore`. La limpieza conserva el original. Excel se omite con un aviso si supera sus límites; los demás formatos se generan.

## Desarrollo y comprobaciones

```shell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -v
python -m pip check
```

GitHub Actions está configurado para repetir las comprobaciones con Python 3.13 en Linux y Windows al publicar el repositorio. Consulta [CONTRIBUTING.md](CONTRIBUTING.md).

## Despliegue

La raíz contiene `app.py`, `requirements.txt` y `.streamlit/config.toml`, listos para seleccionar en Streamlit Community Cloud con **Python 3.13** y rama **main**. La conexión de la cuenta, el remoto y la publicación se completan en una etapa posterior.

Consulta la [guía de despliegue](docs/DEPLOYMENT.md), la [revisión previa a publicación](docs/REVISION.md) y las [notas de los datos de ejemplo](sample_data/README.md).

## Estructura

| Archivo | Responsabilidad |
| --- | --- |
| `app.py` | Interfaz, sesión, secciones y descargas |
| `analysis.py` | Validación, limpieza, auditoría y normalización |
| `data_tools.py` | Lectores, perfilado, hallazgos y reportes |
| `studio_ui.py` | Controles de limpieza, filtros y análisis general |
| `forecasting.py` | Agregación temporal, modelos y evaluación |
| `procesar.py` | Interfaz de consola |
| `test_*.py` | Pruebas del motor, interfaz y regresiones |
| `.github/workflows/ci.yml` | Comprobaciones automáticas |

Los datasets particulares, resultados, entornos virtuales y secretos están excluidos de Git. Los archivos subidos se mantienen en la sesión; no se guardan en el repositorio y deben cargarse de nuevo después de reiniciar la sesión o el servicio.
