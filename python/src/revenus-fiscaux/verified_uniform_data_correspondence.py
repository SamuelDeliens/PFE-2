import pandas as pd
import os
from pathlib import Path
from collections import Counter
import gc
import time


def get_unique_ids(file_paths, id_column='Idcar_200m'):
    """
    Identifie les IDs qui n'apparaissent que dans un seul fichier.

    Args:
        file_paths (list): Liste des chemins vers les fichiers CSV
        id_column (str): Nom de la colonne d'identifiant

    Returns:
        dict: Dictionnaire {fichier: set d'IDs uniques à ce fichier}
    """
    start_time = time.time()

    # Dictionnaire pour stocker tous les IDs avec un compteur
    all_ids_counter = Counter()

    # Dictionnaire pour mapper les IDs à leurs fichiers sources
    id_to_files = {}

    print(f"Lecture des identifiants depuis {len(file_paths)} fichiers...")

    # Première passe: collecter tous les IDs
    for file_path in file_paths:
        file_name = Path(file_path).name
        print(f"Lecture de {file_name}...")

        # Lire uniquement la colonne d'ID pour économiser de la mémoire
        try:
            ids = pd.read_csv(file_path, usecols=[id_column])[id_column].unique()

            # Mettre à jour le compteur
            all_ids_counter.update(ids)

            # Associer chaque ID à son fichier source
            for id_val in ids:
                if id_val not in id_to_files:
                    id_to_files[id_val] = []
                id_to_files[id_val].append(file_name)

            # Libérer la mémoire
            del ids
            gc.collect()

        except Exception as e:
            print(f"Erreur lors de la lecture de {file_name}: {e}")

    # Identifier les IDs uniques (qui n'apparaissent qu'une seule fois)
    unique_ids = {id_val for id_val, count in all_ids_counter.items() if count == 1}
    print(f"Nombre total d'IDs uniques trouvés: {len(unique_ids)}")

    # Organiser les IDs uniques par fichier
    result = {}
    for id_val in unique_ids:
        file_name = id_to_files[id_val][0]  # Un seul fichier par définition
        if file_name not in result:
            result[file_name] = set()
        result[file_name].add(id_val)

    # Afficher les statistiques
    for file_name, ids in result.items():
        print(f"{file_name}: {len(ids)} identifiants uniques")

    print(f"Temps d'exécution: {time.time() - start_time:.2f} secondes")

    return result


def save_results(result, output_dir):
    """
    Sauvegarde les résultats dans des fichiers CSV.

    Args:
        result (dict): Dictionnaire {fichier: set d'IDs uniques}
        output_dir (str): Répertoire de sortie
    """
    os.makedirs(output_dir, exist_ok=True)

    # Créer un fichier de résumé
    with open(os.path.join(output_dir, "resume_ids_uniques.txt"), "w") as f:
        for file_name, ids in result.items():
            f.write(f"{file_name}: {len(ids)} identifiants uniques\n")

    # Créer un fichier CSV par fichier source avec les IDs uniques
    for file_name, ids in result.items():
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}_ids_uniques.csv")

        pd.DataFrame({
            "Idcar_200m": list(ids)
        }).to_csv(output_file, index=False)

        print(f"IDs uniques de {file_name} sauvegardés dans {output_file}")


if __name__ == "__main__":
    dir_path = "./data/revenus-fiscaux/Filosofi/harmonized/"
    output_dir = "./data/revenus-fiscaux/Filosofi/verified/"
    input_files = [dir_path + file for file in os.listdir(dir_path) if file.endswith('.csv')]

    # Identifier les IDs uniques
    result = get_unique_ids(input_files, "Idcar_1km")

    # Sauvegarder les résultats
    save_results(result, output_dir)
