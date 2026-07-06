"""
04_auditar_datos_ogle.py

Auditoría inicial de datos crudos OGLE.

Este script revisa:
- estrellas con banda I y V;
- estrellas válidas con banda I + período;
- estrellas sin banda I;
- estrellas sin período;
- períodos sin curva;
- duplicados por nombre base;
- conteos por proyecto/tipo/campo;
- estrellas que solo tienen banda V.

No modifica archivos.
"""

from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Rutas
# ============================================================

STAR_ROOT = Path("../datos/raw/curvas")
PERIOD_ROOT = Path("../datos/raw/periodos")


# ============================================================
# Utilidades
# ============================================================

def get_project_from_path(path: Path) -> str:
    """Detecta si una ruta pertenece a OGLEIII u OGLEIV."""

    parts = [p.upper() for p in path.parts]

    if "OGLEIII" in parts:
        return "OGLEIII"

    if "OGLEIV" in parts:
        return "OGLEIV"

    raise ValueError(f"No se pudo detectar proyecto en: {path}")


def build_star_id(file_path: Path) -> str:
    """Construye ID único con prefijo de proyecto."""

    project = get_project_from_path(file_path)
    star_name = file_path.name.split(".")[0]

    return f"{project}_{star_name}"


def infer_type_from_name(star_id: str) -> str:
    """Clasifica tipo de estrella desde el nombre."""

    name = star_id.split("_", 1)[1]

    if "RRLYR" in name:
        return "RR Lyrae"
    if "ACEP" in name:
        return "Ceféidas anómalas"
    if "T2CEP" in name:
        return "Ceféidas tipo II"
    if "CEP" in name:
        return "Ceféidas clásicas"

    return "Otros"


def collect_star_ids(files: list[Path]) -> set[str]:
    """Construye conjunto de IDs únicos desde archivos de curvas."""

    star_ids = set()

    for file_path in files:
        try:
            star_ids.add(build_star_id(file_path))
        except ValueError as error:
            print(f"[ADVERTENCIA] {error}")

    return star_ids


def collect_period_ids(period_files: list[Path]) -> set[str]:
    """Construye conjunto de IDs únicos desde archivos de períodos."""

    period_ids = set()

    for file_path in period_files:
        try:
            project = get_project_from_path(file_path)

            with file_path.open("r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    parts = line.split()

                    if len(parts) == 0:
                        continue

                    star_name = parts[0]
                    period_ids.add(f"{project}_{star_name}")

        except ValueError as error:
            print(f"[ADVERTENCIA] {error}")

    return period_ids


# ============================================================
# Auditoría principal
# ============================================================

def main() -> None:

    print("=" * 70)
    print("AUDITORÍA DE DATOS CRUDOS OGLE")
    print("=" * 70)

    if not STAR_ROOT.exists():
        raise FileNotFoundError(f"No existe STAR_ROOT: {STAR_ROOT}")

    if not PERIOD_ROOT.exists():
        raise FileNotFoundError(f"No existe PERIOD_ROOT: {PERIOD_ROOT}")

    # --------------------------------------------------------
    # Archivos de curvas
    # --------------------------------------------------------

    i_files = list(STAR_ROOT.rglob("*.I.dat"))
    v_files = list(STAR_ROOT.rglob("*.V.dat"))
    all_curve_files = list(STAR_ROOT.rglob("*.dat"))

    stars_i = collect_star_ids(i_files)
    stars_v = collect_star_ids(v_files)

    all_stars = stars_i.union(stars_v)

    # --------------------------------------------------------
    # Archivos de períodos
    # --------------------------------------------------------

    period_files = list(PERIOD_ROOT.rglob("*.dat"))
    period_stars = collect_period_ids(period_files)

    # --------------------------------------------------------
    # Diagnósticos generales
    # --------------------------------------------------------

    no_i = all_stars - stars_i
    no_period = all_stars - period_stars
    no_i_and_no_period = no_i.intersection(no_period)
    valid_stars = stars_i.intersection(period_stars)
    missing_curves = period_stars - all_stars

    print("\n--- RESUMEN GENERAL ---")
    print(f"Total estrellas únicas          : {len(all_stars):,}")
    print(f"Con banda I                     : {len(stars_i):,}")
    print(f"Con banda V                     : {len(stars_v):,}")
    print(f"Sin banda I                     : {len(no_i):,}")
    print(f"Sin período                     : {len(no_period):,}")
    print(f"Sin I y sin período             : {len(no_i_and_no_period):,}")
    print(f"Válidas I + período             : {len(valid_stars):,}")
    print(f"Con período pero sin curva      : {len(missing_curves):,}")

    print("\n--- ARCHIVOS ---")
    print(f"Archivos banda I                : {len(i_files):,}")
    print(f"Archivos banda V                : {len(v_files):,}")
    print(f"Archivos de período             : {len(period_files):,}")

    # --------------------------------------------------------
    # Duplicados por nombre base
    # --------------------------------------------------------

    file_names = [file_path.name for file_path in all_curve_files]
    name_counter = Counter(file_names)
    duplicates = {name: n for name, n in name_counter.items() if n > 1}

    print("\n--- DUPLICADOS POR NOMBRE DE ARCHIVO ---")
    print(f"Archivos duplicados             : {len(duplicates):,}")

    for name, count in list(duplicates.items())[:10]:
        print(f"  {name} -> {count}")

    # --------------------------------------------------------
    # Conteos por proyecto/tipo/campo
    # --------------------------------------------------------

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for file_path in i_files:
        parts = file_path.parts

        project = None
        tipo = "UNKNOWN"
        campo = "UNKNOWN"

        for idx, part in enumerate(parts):
            part_upper = part.upper()

            if part_upper in {"OGLEIII", "OGLEIV"}:
                project = part_upper
                tipo = parts[idx + 1] if idx + 1 < len(parts) else "UNKNOWN"
                campo = parts[idx + 2] if idx + 2 < len(parts) else "UNKNOWN"
                break

        if project is None:
            continue

        counts[project][tipo][campo] += 1

    print("\n--- CONTEO POR PROYECTO / TIPO / CAMPO ---")

    total_general = 0

    for project in sorted(counts):
        print(f"\n{project}")
        total_project = 0

        for tipo in sorted(counts[project]):
            total_tipo = 0
            print(f"  {tipo}")

            for campo in sorted(counts[project][tipo]):
                n = counts[project][tipo][campo]
                print(f"    {campo}: {n:,}")
                total_tipo += n

            print(f"    Total {tipo}: {total_tipo:,}")
            total_project += total_tipo

        print(f"  Total {project}: {total_project:,}")
        total_general += total_project

    print(f"\nTOTAL GENERAL CON BANDA I: {total_general:,}")

    # --------------------------------------------------------
    # Estrellas solo V
    # --------------------------------------------------------

    only_v = stars_v - stars_i

    only_v_by_type = Counter()
    only_v_by_project_type = defaultdict(Counter)

    for star_id in only_v:
        project, _ = star_id.split("_", 1)
        tipo = infer_type_from_name(star_id)

        only_v_by_type[tipo] += 1
        only_v_by_project_type[project][tipo] += 1

    print("\n--- ESTRELLAS SOLO BANDA V ---")
    print(f"Total solo V                    : {len(only_v):,}")

    print("\nPor tipo:")
    for tipo, n in only_v_by_type.items():
        print(f"  {tipo}: {n:,}")

    print("\nPor proyecto y tipo:")
    for project in sorted(only_v_by_project_type):
        print(f"\n{project}")
        for tipo, n in only_v_by_project_type[project].items():
            print(f"  {tipo}: {n:,}")


if __name__ == "__main__":
    main()