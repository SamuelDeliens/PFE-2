#!/usr/bin/env python3
import os
import json
import csv
import time
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import pickle
import random

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openaq_data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "17e3ff1f21b0abe58136aedff3f27c50af1adeeab3b1fe561891283e57f600d3"  # Remplacez par votre API key
API_URL = "https://api.openaq.org/v3/sensors/{sensor_id}/hours/monthly"
MAX_THREADS = 10  # Réduit à 2 pour limiter les chances d'erreur too_many_requests
BASE_RATE_LIMIT = 0.1  # Temps d'attente minimal entre les requêtes (secondes)
MAX_RETRIES = 5  # Nombre maximum de tentatives en cas d'erreur
START_YEAR = 2016
END_YEAR = 2023
CHECKPOINT_FILE = "sensor_data_checkpoint.pkl"

# File d'attente pour gérer les délais entre les requêtes globalement
request_timestamps = []
MAX_REQUESTS_PER_MINUTE = 100  # Limite supposée de l'API (à ajuster si nécessaire)


def wait_for_rate_limit():
    """
    Implémente un délai dynamique pour respecter la limite de taux de l'API
    """
    global request_timestamps

    now = time.time()

    # Supprime les timestamps plus vieux que 60 secondes
    request_timestamps = [ts for ts in request_timestamps if now - ts < 60]

    # Si on a atteint la limite de requêtes par minute, attend
    if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        oldest = min(request_timestamps)
        sleep_time = 61 - (now - oldest)
        if sleep_time > 0:
            logger.info(f"Limite de taux atteinte, attente de {sleep_time:.2f} secondes")
            time.sleep(sleep_time)

    # Ajoute un délai de base pour éviter les rafales
    jitter = random.uniform(0, 0.5)  # Ajoute un peu d'aléatoire pour éviter la synchronisation
    time.sleep(BASE_RATE_LIMIT + jitter)

    # Enregistre ce timestamp
    request_timestamps.append(time.time())


def load_checkpoint():
    """
    Charge le point de reprise s'il existe
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du point de reprise: {e}")
    return {}


def save_checkpoint(checkpoint_data):
    """
    Sauvegarde le point de reprise
    """
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du point de reprise: {e}")


def load_sensor_country_mapping(csv_file):
    """
    Charge le fichier CSV contenant la correspondance entre capteurs et pays
    """
    sensor_country_map = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sensor_country_map[int(row['capteurID'])] = row['Pays']
        logger.info(f"Mappings chargés: {len(sensor_country_map)} capteurs")
        return sensor_country_map
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier CSV: {e}")
        return {}


def get_sensor_parameter_info(sensor_id, checkpoint_data):
    """
    Récupère les informations sur le type de paramètre mesuré par le capteur
    """
    # Vérifie si l'information est déjà dans le checkpoint
    if f"param_info_{sensor_id}" in checkpoint_data:
        logger.info(f"Utilisation des infos en cache pour le capteur {sensor_id}")
        return checkpoint_data[f"param_info_{sensor_id}"]

    headers = {
        "accept": "application/json",
        "X-API-Key": API_KEY
    }

    url = f"https://api.openaq.org/v3/sensors/{sensor_id}"

    for attempt in range(MAX_RETRIES):
        try:
            wait_for_rate_limit()
            logger.info(f"Récupération des infos du capteur {sensor_id} (tentative {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, headers=headers)

            # Gestion spécifique de l'erreur too_many_requests
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"Limite de taux dépassée, attente de {retry_after} secondes")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                sensor = data['results'][0]
                parameter_name = sensor['parameter']['name']
                parameter_units = sensor['parameter']['units']
                result = (parameter_name, parameter_units)

                # Sauvegarde dans le checkpoint
                checkpoint_data[f"param_info_{sensor_id}"] = result
                save_checkpoint(checkpoint_data)

                return result
            else:
                logger.warning(f"Aucune information trouvée pour le capteur {sensor_id}")
                result = ("unknown", "unknown")
                checkpoint_data[f"param_info_{sensor_id}"] = result
                save_checkpoint(checkpoint_data)
                return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête pour le capteur {sensor_id}: {e}")
            # Si c'est la dernière tentative, renvoie une valeur par défaut
            if attempt == MAX_RETRIES - 1:
                result = ("unknown", "unknown")
                checkpoint_data[f"param_info_{sensor_id}"] = result
                save_checkpoint(checkpoint_data)
                return result
            # Sinon attend avant de réessayer
            backoff_time = (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"Nouvelle tentative dans {backoff_time:.2f} secondes")
            time.sleep(backoff_time)


def fetch_monthly_data(sensor_id, year, month, checkpoint_data):
    """
    Récupère les données mensuelles pour un capteur et une période donnée
    """
    # Clé unique pour identifier cette requête dans le checkpoint
    request_key = f"data_{sensor_id}_{year}_{month:02d}"

    # Vérifie si les données sont déjà dans le checkpoint
    if request_key in checkpoint_data:
        logger.info(f"Utilisation des données en cache pour {request_key}")
        return checkpoint_data[request_key]

    headers = {
        "accept": "application/json",
        "X-API-Key": API_KEY
    }

    # Formatage des dates pour la requête
    if month == 12:
        from_date = f"{year}-{month:02d}-01T00:00:00Z"
        to_date = f"{year + 1}-01-01T00:00:00Z"
    else:
        from_date = f"{year}-{month:02d}-01T00:00:00Z"
        to_date = f"{year}-{month + 1:02d}-01T00:00:00Z"

    params = {
        "datetime_from": from_date,
        "datetime_to": to_date,
        "limit": 100
    }

    url = API_URL.format(sensor_id=sensor_id)

    for attempt in range(MAX_RETRIES):
        try:
            wait_for_rate_limit()
            logger.info(f"Requête API pour {request_key} (tentative {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, headers=headers, params=params)

            # Gestion spécifique de l'erreur too_many_requests
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"Limite de taux dépassée, attente de {retry_after} secondes")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            # Sauvegarde dans le checkpoint
            checkpoint_data[request_key] = data
            save_checkpoint(checkpoint_data)

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête pour {request_key}: {e}")
            # Si c'est la dernière tentative, renvoie None
            if attempt == MAX_RETRIES - 1:
                checkpoint_data[request_key] = None
                save_checkpoint(checkpoint_data)
                return None
            # Sinon attend avant de réessayer avec backoff exponentiel
            backoff_time = 2
            logger.info(f"Nouvelle tentative dans {backoff_time:.2f} secondes")
            time.sleep(backoff_time)


def process_sensor_data(sensor_id, country):
    """
    Traite un capteur, récupère toutes ses données mensuelles et les sauvegarde
    """
    # Charge le point de reprise
    checkpoint_data = load_checkpoint()

    # Crée le dossier du pays s'il n'existe pas
    country_dir = Path(f"data/{country}")
    country_dir.mkdir(parents=True, exist_ok=True)

    # Récupère le type de paramètre et l'unité
    parameter_name, parameter_units = get_sensor_parameter_info(sensor_id, checkpoint_data)

    # Nom du fichier de sortie
    output_file = country_dir / f"sensor_monthly_{sensor_id}_{parameter_name}.csv"

    # Si le fichier existe déjà, on passe
    if output_file.exists():
        logger.info(f"Le fichier {output_file} existe déjà, passage au suivant")
        return True

    # Fichier temporaire pour sauvegarder les données au fur et à mesure
    temp_file = country_dir / f"temp_{sensor_id}_{parameter_name}.csv"

    # Vérifie si des données temporaires existent déjà
    all_results = []
    if temp_file.exists():
        try:
            df_temp = pd.read_csv(temp_file)
            all_results = df_temp.values.tolist()
            logger.info(f"Chargement de {len(all_results)} résultats temporaires pour le capteur {sensor_id}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier temporaire: {e}")

    # Clé de suivi pour le checkpoint
    processed_dates_key = f"processed_dates_{sensor_id}"
    processed_dates = checkpoint_data.get(processed_dates_key, set())

    # Parcours de toutes les périodes de 2016 à 2023
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            # Vérifie si cette période a déjà été traitée
            period_key = f"{year}-{month:02d}"
            if period_key in processed_dates:
                logger.info(f"Période {period_key} déjà traitée pour le capteur {sensor_id}")
                continue

            data = fetch_monthly_data(sensor_id, year, month, checkpoint_data)

            if not data or 'results' not in data or not data['results']:
                # Marque comme traitée même si pas de données
                processed_dates.add(period_key)
                checkpoint_data[processed_dates_key] = processed_dates
                save_checkpoint(checkpoint_data)
                continue

            new_results_count = 0
            for result in data['results']:
                if 'period' not in result or 'summary' not in result:
                    continue

                try:
                    # Date de début de la période
                    if 'datetimeFrom' in result['period'] and 'utc' in result['period']['datetimeFrom']:
                        date = result['period']['datetimeFrom']['utc']
                    else:
                        continue

                    # Extraction des valeurs
                    value = result.get('value', 'NA')
                    min_val = result['summary'].get('min', 'NA')
                    q02 = result['summary'].get('q02', 'NA')
                    q25 = result['summary'].get('q25', 'NA')
                    median = result['summary'].get('median', 'NA')
                    q75 = result['summary'].get('q75', 'NA')
                    q98 = result['summary'].get('q98', 'NA')
                    max_val = result['summary'].get('max', 'NA')
                    avg = result['summary'].get('avg', 'NA')
                    sd = result['summary'].get('sd', 'NA')

                    all_results.append([
                        date, value, min_val, q02, q25, median, q75, q98, max_val, avg, sd
                    ])
                    new_results_count += 1
                except Exception as e:
                    logger.error(f"Erreur lors du traitement des données: {e}")

            # Sauvegarde temporaire après chaque mois traité
            if new_results_count > 0:
                logger.info(f"Ajout de {new_results_count} nouveaux résultats pour {period_key}")
                try:
                    df = pd.DataFrame(all_results, columns=[
                        'date', 'value', 'min', 'q02', 'q25', 'median', 'q75', 'q98', 'max', 'avg', 'sd'
                    ])
                    df.to_csv(temp_file, index=False)
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde temporaire: {e}")

            # Marque cette période comme traitée
            processed_dates.add(period_key)
            checkpoint_data[processed_dates_key] = processed_dates
            save_checkpoint(checkpoint_data)

    # Sauvegarde finale des données dans un fichier CSV
    if all_results:
        try:
            # Crée un DataFrame pour faciliter le tri et l'écriture
            df = pd.DataFrame(all_results, columns=[
                'date', 'value', 'min', 'q02', 'q25', 'median', 'q75', 'q98', 'max', 'avg', 'sd'
            ])

            # Supprime les doublons éventuels
            df = df.drop_duplicates(subset=['date'])

            # Tri par date
            df = df.sort_values('date')

            # Écriture dans le fichier CSV final
            df.to_csv(output_file, index=False)

            logger.info(f"Données pour le capteur {sensor_id} ({parameter_name}) sauvegardées dans {output_file}")

            # Supprime le fichier temporaire
            if temp_file.exists():
                temp_file.unlink()

            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données pour le capteur {sensor_id}: {e}")
            return False
    else:
        logger.warning(f"Aucune donnée trouvée pour le capteur {sensor_id}")
        return True


def main():
    """
    Fonction principale qui coordonne tout le processus
    """
    # Chemin vers le fichier CSV contenant la correspondance capteur-pays
    csv_file = "./data/temps/sensor_countries.csv"

    # Charge les mappings entre capteurs et pays
    sensor_country_map = load_sensor_country_mapping(csv_file)

    if not sensor_country_map:
        logger.error("Aucun mapping capteur-pays trouvé. Arrêt du script.")
        return

    # Charge l'état de progression
    checkpoint_data = load_checkpoint()
    processed_sensors = checkpoint_data.get('completed_sensors', set())

    # Filtrer les capteurs déjà traités
    sensors_to_process = {sensor_id: country for sensor_id, country in sensor_country_map.items()
                          if sensor_id not in processed_sensors}

    # Nombre total de capteurs à traiter
    total_sensors = len(sensors_to_process)
    logger.info(f"Traitement de {total_sensors} capteurs restants...")

    # Si tous les capteurs ont été traités
    if total_sensors == 0:
        logger.info("Tous les capteurs ont déjà été traités!")
        return

    # Crée le dossier de données s'il n'existe pas
    Path("data2").mkdir(exist_ok=True)

    # Traite les capteurs en séquence pour mieux gérer les limites de taux
    completed = 0
    for sensor_id, country in sensors_to_process.items():
        logger.info(f"Traitement du capteur {sensor_id} pour {country}")
        success = process_sensor_data(sensor_id, country)

        if success:
            # Marque ce capteur comme traité
            processed_sensors.add(sensor_id)
            checkpoint_data['completed_sensors'] = processed_sensors
            save_checkpoint(checkpoint_data)

            completed += 1
            logger.info(
                f"Progression: {completed}/{total_sensors} capteurs traités ({completed / total_sensors * 100:.2f}%)")

    logger.info("Traitement terminé! Toutes les données ont été récupérées.")


if __name__ == "__main__":
    main()