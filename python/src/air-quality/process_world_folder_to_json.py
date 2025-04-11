import os
import json
import csv
import re
import pandas as pd
import numpy as np
from datetime import datetime


# Fonction pour extraire le type de polluant depuis le nom du fichier
def extract_pollutant_type(filename):
    pollutant_types = ["pm25", "pm10", "o3", "so2", "co", "no2"]
    for pollutant in pollutant_types:
        if filename.endswith(f"_{pollutant}.csv"):
            return pollutant
    return None


def collect_sensor_data(root_directory, output_json_file="pollutant_data.json"):
    """
    Collecte les données des capteurs et les organise en structure JSON.

    Args:
        root_directory (str): Chemin du dossier principal contenant les dossiers des pays
        output_json_file (str): Chemin pour le fichier JSON de sortie

    Returns:
        dict: Données collectées au format JSON
    """
    print(f"Collecte des données à partir du dossier: {root_directory}")

    # Dictionnaire pour stocker les données
    data = {}

    # Parcourir tous les dossiers des pays
    country_folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]
    print(f"Pays trouvés: {len(country_folders)}")

    for country_folder in country_folders:
        country_path = os.path.join(root_directory, country_folder)
        data[country_folder] = {}

        # Parcourir tous les fichiers de capteurs dans le dossier du pays
        sensor_files = [f for f in os.listdir(country_path) if f.startswith("sensor_monthly_") and f.endswith(".csv")]
        print(f"Traitement de {country_folder} - {len(sensor_files)} fichiers de capteurs trouvés")

        for sensor_file in sensor_files:
            pollutant_type = extract_pollutant_type(sensor_file)
            if not pollutant_type:
                continue

            file_path = os.path.join(country_path, sensor_file)

            # Lire le fichier CSV
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)

                for row in csv_reader:
                    date = row['date']

                    # S'assurer que la date est au format YYYY-MM
                    if len(date) > 7:
                        date = date[:7]  # Prendre seulement YYYY-MM

                    # Créer la structure si elle n'existe pas encore
                    if date not in data[country_folder]:
                        data[country_folder][date] = {}

                    # Convertir les valeurs en nombres
                    numeric_data = {}
                    for key in ['value', 'min', 'q02', 'q25', 'median', 'q75', 'q98', 'max', 'avg', 'sd']:
                        try:
                            numeric_data[key] = float(row[key])
                        except (ValueError, KeyError):
                            numeric_data[key] = None

                    # Stocker les données pour ce polluant et cette date
                    data[country_folder][date][pollutant_type] = numeric_data

    # Écrire les données dans un fichier JSON
    with open(output_json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2)

    print(f"Données collectées et enregistrées dans {output_json_file}")
    return data


def fill_missing_months(data, output_json_file=None):
    """
    Identifie et remplit les mois manquants dans les données de capteurs.

    Args:
        data (dict): Données des capteurs au format JSON
        output_json_file (str, optional): Chemin pour le fichier JSON de sortie.
                                         Si None, pas de sauvegarde dans un fichier.

    Returns:
        dict: Données avec les mois manquants remplis
    """
    print("Début du remplissage des mois manquants...")

    # Pour chaque pays dans les données
    for country, dates in data.items():
        print(f"Traitement du pays: {country}")

        # Si le pays n'a pas de données, passer au suivant
        if not dates:
            continue

        # Convertir les dates en objets datetime pour le tri
        date_objects = []
        for date_str in dates.keys():
            try:
                # Assumer un format YYYY-MM ou YYYY-MM-DD
                if len(date_str) == 7:  # YYYY-MM
                    date_obj = datetime.strptime(date_str, '%Y-%m')
                else:  # YYYY-MM-DD ou autre format
                    date_obj = datetime.strptime(date_str[:7], '%Y-%m')
                date_objects.append((date_str, date_obj))
            except ValueError:
                print(f"  Format de date non reconnu: {date_str}, ignoré")

        # Si moins de 2 dates, impossible d'interpoler
        if len(date_objects) < 2:
            print(f"  Pas assez de données pour {country} pour interpoler")
            continue

        # Trier les dates
        date_objects.sort(key=lambda x: x[1])

        # Trouver la première et dernière date
        first_date = date_objects[0][1]
        last_date = date_objects[-1][1]

        # Créer une liste de toutes les dates mensuelles entre la première et la dernière
        all_months = []
        current_date = first_date
        while current_date <= last_date:
            all_months.append(current_date.strftime('%Y-%m'))
            # Passer au mois suivant
            year = current_date.year + (current_date.month // 12)
            month = (current_date.month % 12) + 1
            current_date = datetime(year, month, 1)

        # Identifier les mois manquants
        existing_months = [date_str[:7] if len(date_str) > 7 else date_str for date_str, _ in date_objects]
        missing_months = [month for month in all_months if month not in existing_months]

        if missing_months:
            print(f"  Mois manquants pour {country}: {len(missing_months)} mois")

            # Pour chaque type de polluant, traiter séparément
            pollutant_types = set()
            for date_data in dates.values():
                pollutant_types.update(date_data.keys())

            for pollutant in pollutant_types:
                # Créer un DataFrame pour ce polluant
                df = pd.DataFrame(index=existing_months)

                # Collecter toutes les mesures disponibles pour ce polluant
                for date_str in existing_months:
                    # Trouver la clé correspondante dans les données originales
                    original_key = next((k for k in dates.keys() if k.startswith(date_str)), None)
                    if original_key and pollutant in dates[original_key]:
                        for measure, value in dates[original_key][pollutant].items():
                            if measure not in df.columns:
                                df[measure] = None
                            df.loc[date_str, measure] = value

                # Convertir en valeurs numériques
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Réindexer avec tous les mois
                df = df.reindex(all_months)

                # Remplir les mois manquants
                for month in missing_months:
                    month_date = datetime.strptime(month, '%Y-%m')

                    # Stratégie 1: Réutiliser les valeurs du même mois de l'année précédente/suivante
                    same_month_prev_year = f"{month_date.year - 1}-{month_date.month:02d}"
                    same_month_next_year = f"{month_date.year + 1}-{month_date.month:02d}"

                    if same_month_prev_year in existing_months:
                        print(f"    {month} ({pollutant}): Utilisation des données du même mois de l'année précédente")
                        df.loc[month] = df.loc[same_month_prev_year]
                        continue

                    if same_month_next_year in existing_months:
                        print(f"    {month} ({pollutant}): Utilisation des données du même mois de l'année suivante")
                        df.loc[month] = df.loc[same_month_next_year]
                        continue

                    # Stratégie 2: Interpolation linéaire
                    try:
                        # Vérifier s'il y a des valeurs non-NaN avant et après le mois actuel
                        before = df.loc[:month].dropna(how='all')
                        after = df.loc[month:].dropna(how='all')

                        if not before.empty and not after.empty:
                            print(f"    {month} ({pollutant}): Interpolation linéaire")
                            # Interpolation linéaire
                            df_interpolated = df.interpolate(method='linear')
                            df.loc[month] = df_interpolated.loc[month]
                            continue
                    except Exception as e:
                        print(f"    Erreur lors de l'interpolation pour {month} ({pollutant}): {e}")

                    # Stratégie 3: Utiliser la valeur du mois le plus proche
                    print(f"    {month} ({pollutant}): Utilisation du mois le plus proche")

                    # Trouver le mois existant le plus proche
                    month_date_ts = pd.Timestamp(month_date)
                    existing_dates = [pd.Timestamp(datetime.strptime(x, '%Y-%m')) for x in existing_months]
                    closest_idx = np.argmin([abs((date - month_date_ts).days) for date in existing_dates])
                    closest_month = existing_months[closest_idx]

                    df.loc[month] = df.loc[closest_month]

                # Mettre à jour les données avec les valeurs interpolées
                for month in all_months:
                    if month not in dates:
                        dates[month] = {}

                    # Convertir les valeurs en dictionnaire sans NaN
                    values = df.loc[month].to_dict()
                    clean_values = {k: float(v) for k, v in values.items() if not pd.isna(v)}

                    if clean_values:  # Si des valeurs existent
                        if pollutant not in dates[month]:
                            dates[month][pollutant] = {}
                        dates[month][pollutant] = clean_values

    # Écrire le résultat en JSON si demandé
    if output_json_file:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Traitement terminé. Fichier {output_json_file} créé/mis à jour.")

    return data


def calculate_composite_index(data):
    """
    Calcule un indice composite pour chaque pays et chaque mois en combinant
    les différents types de polluants.

    Args:
        data (dict): Données des capteurs avec tous les mois remplis

    Returns:
        dict: Données avec l'indice composite ajouté sous le type 'composite'
    """
    print("Calcul de l'indice composite des polluants...")

    # Poids relatifs pour chaque polluant (basés sur des standards internationaux approximatifs)
    # Ces poids peuvent être ajustés selon vos besoins spécifiques
    weights = {
        "pm25": 1.0,  # Particules fines PM2.5
        "pm10": 0.5,  # Particules PM10
        "o3": 0.7,  # Ozone
        "no2": 0.6,  # Dioxyde d'azote
        "so2": 0.4,  # Dioxyde de soufre
        "co": 0.3  # Monoxyde de carbone
    }

    # Pour chaque pays
    for country, dates in data.items():
        print(f"Calcul de l'indice composite pour: {country}")

        # Pour chaque date
        for date, pollutants in dates.items():
            # Vérifier s'il y a des données de polluants pour cette date
            if not pollutants:
                continue

            # Variables pour le calcul de l'indice composite
            weighted_sum = 0
            total_weight = 0
            pollutant_count = 0

            # Pour chaque mesure disponible, utiliser la valeur 'value' comme référence
            for pollutant_type, measurements in pollutants.items():
                if pollutant_type in weights and 'value' in measurements:
                    value = measurements['value']
                    if value is not None:
                        weighted_sum += value * weights[pollutant_type]
                        total_weight += weights[pollutant_type]
                        pollutant_count += 1

            # Si des données sont disponibles, calculer l'indice composite
            if pollutant_count > 0 and total_weight > 0:
                # Calculer la moyenne pondérée
                composite_value = weighted_sum / total_weight

                # Créer un dictionnaire pour l'indice composite avec les mêmes métriques
                composite_data = {
                    'value': composite_value,
                    'min': composite_value * 0.8,  # Approximation
                    'max': composite_value * 1.2,  # Approximation
                    'avg': composite_value,
                    'sd': composite_value * 0.1,  # Approximation
                    'median': composite_value,
                    'q25': composite_value * 0.9,  # Approximation
                    'q75': composite_value * 1.1,  # Approximation
                    'q02': composite_value * 0.7,  # Approximation
                    'q98': composite_value * 1.3  # Approximation
                }

                # Ajouter l'indice composite aux données
                pollutants['composite'] = composite_data

    return data


def process_sensor_data(root_directory, output_json_file="pollutant_data_completed.json"):
    """
    Fonction principale qui collecte les données, remplit les mois manquants,
    et calcule l'indice composite des polluants.

    Args:
        root_directory (str): Chemin du dossier principal contenant les dossiers des pays
        output_json_file (str): Chemin pour le fichier JSON final

    Returns:
        dict: Données complètes avec les mois manquants remplis et l'indice composite
    """
    # Étape 1: Collecter les données
    tmp_json = "temp_collected_data.json"
    data = collect_sensor_data(root_directory, tmp_json)

    # Étape 2: Remplir les mois manquants
    completed_data = fill_missing_months(data)

    # Étape 3: Calculer l'indice composite
    final_data = calculate_composite_index(completed_data)

    # Enregistrer les données finales
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"Traitement terminé. Fichier {output_json_file} créé/mis à jour.")

    # Nettoyage du fichier temporaire
    if os.path.exists(tmp_json):
        os.remove(tmp_json)
        print(f"Fichier temporaire {tmp_json} supprimé")

    return final_data


if __name__ == "__main__":
    # Chemin du dossier principal contenant les dossiers des pays
    root_directory = "./data/air-quality/world"  # À remplacer par votre chemin

    # Fichier de sortie
    output_file = "./data/air-quality/pollutant_data_completed.json"

    # Exécuter le traitement complet
    process_sensor_data(root_directory, output_file)
