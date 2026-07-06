from pathlib import Path

base = Path("../datos/raw")

eliminados = 0

print(f"Se limpiarán los archivos de: {base.resolve()}")

respuesta = input("¿Continuar? [s/N]: ")

if respuesta.lower() != "s":
    print("Operación cancelada.")
    raise SystemExit

for archivo in base.rglob("*"):

    if not archivo.is_file():
        continue

    nombre = archivo.name

    if nombre.endswith(".I.dat") or nombre.endswith(".V.dat"):
        continue

    archivo.unlink()
    eliminados += 1

print(f"Archivos eliminados: {eliminados}")


