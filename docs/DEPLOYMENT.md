# Preparación para Streamlit Community Cloud

El repositorio está preparado para alojarse con el nombre **dataclara-analytics**. La aplicación abre en la carga de archivos y calcula los resultados a partir del dataset de la sesión.

## Configuración prevista

| Campo | Valor |
| --- | --- |
| Rama | `main` |
| Archivo principal | `app.py` |
| Python | `3.13` |
| Dependencias | `requirements.txt` |
| Tema y límite de carga | `.streamlit/config.toml` |

Community Cloud ejecuta desde la raíz del repositorio. Por eso `app.py`, `requirements.txt` y `.streamlit/config.toml` están en esa raíz. La versión de Python se debe elegir en **Advanced settings** al crear el despliegue; `.python-version` también documenta la versión local. Consulta la [organización de archivos](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization) y la [configuración del despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Conectar en la siguiente etapa

1. Crear o seleccionar el repositorio de GitHub con el nombre indicado y la visibilidad deseada.
2. Configurar su URL como remoto `origin` y publicar `main`.
3. Revisar que las comprobaciones automáticas terminen correctamente.
4. Conectar la cuenta de GitHub en Streamlit Community Cloud.
5. Elegir el repositorio, `main`, `app.py` y Python 3.13; después desplegar.

Esta preparación no configura un remoto ni publica archivos o credenciales. Las acciones de CI están preparadas; su primera ejecución en GitHub ocurrirá al publicar el repositorio.

## Datos y recursos

- La subida de archivos está limitada a 50 MB por archivo.
- Los datos subidos y resultados se mantienen en la sesión. No se escriben en el repositorio. Al reiniciar la sesión o el servicio hay que cargar los archivos de nuevo.
- `.gitignore` excluye datasets particulares, resultados, entornos, cachés y secretos. Solo se incluyen dos pequeñas muestras que usan las pruebas y las demostraciones.
- Excel tiene un límite preventivo de 1.000.000 de celdas entre todas las hojas. Si se supera, siguen disponibles CSV, JSON, Parquet e informe HTML.
- HTML muestra hasta 5.000 registros de auditoría e informa cuando la vista está limitada. La descarga CSV conserva la auditoría seleccionada completa.
- La auditoría automática resume las operaciones en datasets de más de 50.000 filas. El detalle completo puede consumir bastante memoria.

Las versiones directas están fijadas para facilitar la reproducción. Las dependencias transitivas se resuelven al instalar. La configuración de CI comprueba instalación, Ruff y pruebas en Linux y Windows. El despliegue real y el rendimiento bajo concurrencia se verifican después de conectar la cuenta; aún no se han ejecutado en Community Cloud.
