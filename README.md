# DataClara · Minería de datos

Aplicación de Python y Streamlit que calcula sus resultados con el dataset cargado por el usuario.

## Ejecutar

Requiere Python 3.13. Desde esta carpeta:

```shell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Windows también puedes abrir `iniciar.bat`.

## Uso

1. Sube un CSV o Excel (.xlsx) de hasta 50 MB.
2. Revisa las columnas numéricas, selecciona fechas y confirma sus formatos.
3. Ajusta las reglas y pulsa **Aplicar limpieza**.
4. En **Limpieza**, compara las vistas antes/después, revisa el registro resumido y prepara la descarga CSV.
5. En **Normalización**, aplica Min–Max o Z-score sobre una copia y descarga el resultado.
6. En **Análisis**, consulta productos más vendidos, importes netos, devoluciones, barras por categoría, distribución, dispersión y correlación. El resumen de productos permite seleccionar sus columnas y descargar el ranking completo.
7. En **Predicción**, elige fecha, métrica, frecuencia y modelo para evaluar y proyectar escenarios.

Los espacios y marcadores de valores vacíos se uniforman automáticamente. Puedes eliminar duplicados, rellenar faltantes con mediana/texto, eliminar filas incompletas, quitar columnas por proporción de nulos o recortar atípicos con IQR. Los identificadores deben conservarse como texto. Las fechas ambiguas requieren confirmar día y mes.

Las fechas reconocibles se preseleccionan mediante una muestra distribuida del archivo. En barras, se priorizan columnas con pocas categorías y los códigos se muestran como etiquetas, sin convertirlos en un eje numérico.

La lectura y limpieza se reutilizan durante la sesión. Cambiar de sección o de gráfica no repite la limpieza. Las vistas muestran hasta 100 filas y la dispersión hasta 5.000 puntos; las descargas conservan todas las filas de su resultado.

Las proyecciones requieren al menos 16 periodos regulares y se evalúan con separación temporal de entrenamiento, validación y prueba. La banda es orientativa; los escenarios son hipótesis y no garantizan resultados futuros. Si se detectan Quantity y UnitPrice, se ofrece su producto como importe, incluyendo movimientos negativos.

## Archivos

| Archivo | Función |
| --- | --- |
| `app.py` | Carga, controles, cuatro secciones y descargas CSV |
| `analysis.py` | Lectura, limpieza, calidad y normalización |
| `forecasting.py` | Series temporales, modelos, evaluación y gráfica |

Los datasets locales se conservan como archivos del usuario; no se cargan automáticamente.

## Streamlit Community Cloud

Repositorio: `arcelot27/trabajo2-streamlit-mineria-de-datos`. Rama: `main`. Archivo principal: `app.py`. Selecciona Python **3.13** en **Advanced settings**. Las dependencias están en `requirements.txt` y el tema oscuro en `.streamlit/config.toml`.

Al subir cambios a `main` en GitHub, Community Cloud actualiza la aplicación automáticamente. Los cambios en dependencias pueden tardar unos minutos. Puedes revisar el estado desde **Manage app** y reiniciar desde el menú **Reboot app** si es necesario.

# Elaborado por:
## Arturo Castañeda Murcia
## Alan Zapata Torres
