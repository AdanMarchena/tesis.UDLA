"""
Extrae automáticamente todos los archivos .tar descargados desde OGLE.

Cada archivo se extrae en una carpeta con el mismo nombre y posteriormente
el archivo comprimido se elimina.

Script auxiliar utilizado durante la construcción del dataset.
"""

from pathlib import Path
import tarfile

ROOT = Path("../datos/raw/curvas")

tar_files = list(ROOT.rglob("*.tar"))

print(f"Se encontraron {len(tar_files)} archivos .tar.")

extraidos = 0

for tar_path in tar_files:

    extract_folder = tar_path.with_suffix("")
    extract_folder.mkdir(exist_ok=True)

    print(f"Extrayendo: {tar_path.name}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_folder)

    tar_path.unlink()

    extraidos += 1

print("\nProceso finalizado.")
print(f"Archivos extraídos: {extraidos}")