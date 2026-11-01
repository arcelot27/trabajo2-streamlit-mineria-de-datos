# Contribuir a DataClara

Usa Python 3.13 y un entorno virtual. Instala las herramientas de desarrollo:

```shell
python -m pip install -r requirements-dev.txt
```

Antes de crear un commit:

```shell
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -v
python -m pip check
git diff --check
```

Para aplicar el formato ejecuta `python -m ruff format .`. Agrega pruebas cuando corrijas un comportamiento observable o un fallo de datos. Los cambios de datos deben conservar el archivo original y describir cómo se modifica la auditoría.

Trabaja en una rama con un nombre descriptivo, por ejemplo `fix/validacion-csv`. Usa commits breves que expliquen el cambio: `fix: rechazar filas CSV con campos adicionales`.

Los datasets de usuarios, exportaciones, entornos virtuales, credenciales y archivos de configuración privada no se versionan. Revisa `git status --short` y el contenido preparado antes del commit. Las credenciales de un despliegue se configuran en Streamlit Community Cloud, nunca en el código.

Al actualizar dependencias, conserva los pines probados, ejecuta las comprobaciones y revisa el resultado de CI en Linux y Windows. Una prueba local en Windows no confirma por sí sola el despliegue en Linux.
