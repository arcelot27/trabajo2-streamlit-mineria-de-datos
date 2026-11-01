# Revisión previa a publicación

Fecha: 31 de octubre de 2026. Alcance: motor de datos, predicción, sesión de Streamlit, exportaciones, dependencias y archivos que formarán el primer commit.

## Defectos corregidos

| Problema confirmado | Cambio y evidencia |
| --- | --- |
| CSV con campos adicionales en la primera fila podía interpretarse como índice implícito y perder información | `analysis.py`, función `load_csv`: índice automático desactivado y advertencias de truncamiento convertidas en errores. Prueba: `test_csv_rejects_implicit_index_and_truncated_fields`. |
| Al cargar un segundo archivo quedaban exportaciones y copias normalizadas del anterior en memoria | `app.py`, carga de un archivo nuevo: invalidación de cachés derivadas después de una lectura válida. Prueba: `test_new_upload_discards_previous_heavy_exports`. |
| Excel grande podía superar la memoria del servicio antes de comprobar el tamaño | `data_tools.py`, `build_excel`: validación previa del número de celdas y de los límites de cada hoja. Prueba: `test_excel_limit_runs_before_allocating_workbook`. |
| Un rechazo de Excel impedía obtener los demás formatos | `app.py` y `procesar.py`: se omite Excel con un aviso, conservando HTML, Parquet y las descargas CSV/JSON. Prueba: `test_large_excel_does_not_block_html_or_parquet`. |
| HTML podía materializar millones de registros de auditoría en una sola página | `data_tools.py`, `build_html_report`: vista de auditoría limitada a 5.000 filas con nota y descarga completa separada. Prueba: `test_html_limits_audit_and_keeps_complete_download_note`. |
| Series con fechas irregulares podían recibir fechas futuras diarias por defecto | `forecasting.py`, `forecast`: validación de índice temporal y frecuencia regular. Prueba: `test_forecast_rejects_irregular_periods_and_non_dates`. |
| Pruebas de interfaz dependían del orden de los archivos propios del usuario | `test_analysis.py` y `test_studio.py`: la fuente local de las pruebas se fija a su muestra. Los datasets particulares dejan de cambiar el resultado de las pruebas. |

## Preparación del repositorio

- Dependencias directas fijadas a las versiones probadas y Python 3.13 documentado.
- Ruff configurado para errores, imports y patrones propensos a fallos; formato uniforme.
- GitHub Actions con permisos de lectura, tiempo máximo y acciones fijadas por identificador de commit; matriz Linux/Windows.
- Dependabot configurado para proponer actualizaciones de Python y GitHub Actions.
- Exclusiones de datasets particulares, resultados, cachés, entornos, secretos y notas del equipo. Solo se versionan dos muestras pequeñas.
- README, guía de despliegue y guía de contribución con rutas relativas y comandos reproducibles.

## Verificación y límites

Las comprobaciones locales incluyen Ruff, formato, 36 pruebas unitarias e interacción con AppTest, validación de dependencias y ejecución de la aplicación con AppTest desde una copia que contiene únicamente los archivos preparados para Git.

Las pruebas locales se ejecutan en Windows con Python 3.13. El flujo para Linux está preparado en CI, pero no se ejecutará en GitHub hasta conectar y publicar el repositorio. Tampoco se ha desplegado todavía en Streamlit Community Cloud. Esta revisión no afirma cobertura de todas las combinaciones de archivos ni sustituye la verificación del despliegue real.
