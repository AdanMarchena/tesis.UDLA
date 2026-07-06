from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIODOS_DIR = PROJECT_ROOT / "datos" / "raw" / "periodos"
DISCRETIZADOS_DIR = PROJECT_ROOT / "experimentos" / "datos_discretizados"
OUTPUT_DIR = PROJECT_ROOT / "experimentos" / "resultados" / "catalogos"
OUTPUT_PATH = OUTPUT_DIR / "rrlyrae_subtipos.csv"
CATALOGO_FINAL_PATH = PROJECT_ROOT / "experimentos" / "00_datos_limpios" / "catalogo_final.csv"


PATRON_SUBTIPO = re.compile(r"^(a?RR[a-z0-9]+)$", re.IGNORECASE)
PATRON_PREFIJO = re.compile(r"^(OGLEIII|OGLEIV)_(.+)$")


def normalizar_subtipo(nombre_archivo):
    stem = Path(nombre_archivo).stem
    if stem.lower() == "rrlyr":
        return None

    match = PATRON_SUBTIPO.match(stem)
    if not match:
        return None

    subtipo = match.group(1)
    if subtipo.lower().startswith("arr"):
        return "aRR" + subtipo[3:]
    return "RR" + subtipo[2:]


def quitar_prefijo(star_id):
    match = PATRON_PREFIJO.match(str(star_id))
    return match.group(2) if match else str(star_id)


def leer_ids_periodo(ruta):
    filas = []
    with open(ruta, "r", encoding="utf-8", errors="ignore") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if partes:
                filas.append(partes[0])
    return filas


def construir_catalogo():
    registros = []

    for ruta in sorted(PERIODOS_DIR.rglob("*.dat")):
        subtipo = normalizar_subtipo(ruta.name)
        if subtipo is None:
            continue

        partes_ruta = ruta.relative_to(PERIODOS_DIR).parts
        try:
            proyecto = partes_ruta[0]
        except IndexError:
            proyecto = None
        region = partes_ruta[1] if len(partes_ruta) > 1 else None

        for star_id_base in leer_ids_periodo(ruta):
            star_id_base = quitar_prefijo(star_id_base)
            star_id = f"{proyecto}_{star_id_base}" if proyecto else pd.NA
            registros.append(
                {
                    "star_id_base": star_id_base,
                    "star_id": star_id,
                    "subtipo_rrlyrae": subtipo,
                    "archivo_origen": str(ruta.relative_to(PROJECT_ROOT)),
                    "proyecto_origen": proyecto,
                    "region_origen": region,
                }
            )

    catalogo = pd.DataFrame(registros)
    if catalogo.empty:
        return catalogo

    catalogo = catalogo.drop_duplicates().sort_values(
        ["subtipo_rrlyrae", "star_id", "archivo_origen"]
    )
    catalogo = agregar_catalogo_final(catalogo)
    return catalogo


def agregar_catalogo_final(catalogo):
    if not CATALOGO_FINAL_PATH.exists():
        catalogo["en_catalogo_final"] = False
        return catalogo

    columnas = ["proyecto", "region", "tipo", "base", "n_obs", "periodo"]
    catalogo_final = pd.read_csv(CATALOGO_FINAL_PATH, usecols=columnas)
    catalogo_final = catalogo_final.rename(
        columns={
            "proyecto": "proyecto_origen",
            "region": "region_catalogo",
            "tipo": "tipo_catalogo",
            "base": "star_id_base",
            "periodo": "periodo_catalogo",
        }
    )
    catalogo_final = catalogo_final.drop_duplicates(
        subset=["proyecto_origen", "star_id_base"]
    )

    enriquecido = catalogo.merge(
        catalogo_final,
        on=["proyecto_origen", "star_id_base"],
        how="left",
    )
    enriquecido["en_catalogo_final"] = enriquecido["tipo_catalogo"].notna()
    return enriquecido


def resumen_matches(catalogo):
    resultados = []
    claves_catalogo = set(catalogo["star_id"].astype(str))
    claves_base_catalogo = set(catalogo["star_id_base"].astype(str))

    for ruta in sorted(DISCRETIZADOS_DIR.rglob("rrlyrae_*.parquet")):
        df = pd.read_parquet(ruta, columns=["star_id"])
        ids = df["star_id"].astype(str)
        ids_base = ids.map(quitar_prefijo)
        match_prefijo = ids.isin(claves_catalogo)
        match_base = ids_base.isin(claves_base_catalogo)
        match_total = match_prefijo | match_base

        resultados.append(
            {
                "dataset": str(ruta.relative_to(PROJECT_ROOT)),
                "n_estrellas": int(len(ids)),
                "match_con_subtipo": int(match_total.sum()),
                "sin_subtipo": int((~match_total).sum()),
                "match_por_star_id_prefijado": int(match_prefijo.sum()),
                "match_por_star_id_base": int(match_base.sum()),
            }
        )

    return pd.DataFrame(resultados)


def main():
    catalogo = construir_catalogo()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalogo.to_csv(OUTPUT_PATH, index=False)

    conteo_subtipos = catalogo["subtipo_rrlyrae"].value_counts().sort_index()
    duplicados_base = int(catalogo["star_id_base"].duplicated(keep=False).sum())
    duplicados_star_id = int(catalogo["star_id"].duplicated(keep=False).sum())
    match_catalogo_final = int(catalogo.get("en_catalogo_final", pd.Series(False)).sum())
    matches = resumen_matches(catalogo)

    print(f"Archivo guardado: {OUTPUT_PATH}")
    print("\nCantidad por subtipo:")
    print(conteo_subtipos.to_string())
    print(f"\nIDs duplicados star_id_base: {duplicados_base}")
    print(f"IDs duplicados star_id: {duplicados_star_id}")
    print(f"Filas con match en catalogo_final.csv: {match_catalogo_final}")
    print("\nMatches datasets discretizados:")
    print(matches.to_string(index=False))


if __name__ == "__main__":
    main()
