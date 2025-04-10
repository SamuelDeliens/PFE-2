import json
import pandas as pd
import numpy as np
from datetime import datetime


def fill_missing_months(input_json_file, output_json_file=None):
    """
    Charge un fichier JSON existant contenant des données de capteurs par pays et par mois,
    identifie et remplit les mois manquants, puis sauvegarde le résultat.

    Args:
        input_json_file (str): Chemin vers le fichier JSON d'entrée
        output_json_file (str, optional): Chemin pour le fichier JSON de sortie.
                                         Si None, remplace le fichier d'entrée.
    """
    if output_json_file is None:
        output_json_file = input_json_file

    # Charger les données JSON existantes
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Fichier JSON chargé: {input_json_file}")

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

            # Créer un DataFrame pour faciliter l'interpolation
            df = pd.DataFrame(index=existing_months)

            # Ajouter toutes les valeurs au DataFrame
            fields = set()  # Pour collecter tous les champs
            for date_str in existing_months:
                # Trouver la clé correspondante dans les données originales
                original_key = next((k for k in dates.keys() if k.startswith(date_str)), None)
                if original_key:
                    for field, value in dates[original_key].items():
                        fields.add(field)
                        if field not in df.columns:
                            df[field] = None
                        df.loc[date_str, field] = value

            # S'assurer que toutes les colonnes sont numériques
            for field in fields:
                df[field] = pd.to_numeric(df[field], errors='coerce')

            # Réindexer avec tous les mois (existants et manquants)
            df = df.reindex(all_months)

            # Stratégies de remplissage pour les mois manquants
            for month in missing_months:
                month_date = datetime.strptime(month, '%Y-%m')

                # Stratégie 1: Réutiliser les valeurs du même mois de l'année précédente/suivante si disponible
                same_month_prev_year = f"{month_date.year - 1}-{month_date.month:02d}"
                same_month_next_year = f"{month_date.year + 1}-{month_date.month:02d}"

                if same_month_prev_year in existing_months:
                    print(f"    {month}: Utilisation des données du même mois de l'année précédente")
                    df.loc[month] = df.loc[same_month_prev_year]
                    continue

                if same_month_next_year in existing_months:
                    print(f"    {month}: Utilisation des données du même mois de l'année suivante")
                    df.loc[month] = df.loc[same_month_next_year]
                    continue

                # Stratégie 2: Interpolation linéaire si le mois est entre deux mois existants
                try:
                    # Vérifier s'il y a des valeurs non-NaN avant et après le mois actuel
                    before = df.loc[:month].dropna(how='all')
                    after = df.loc[month:].dropna(how='all')

                    if not before.empty and not after.empty:
                        print(f"    {month}: Interpolation linéaire")
                        # Interpolation linéaire
                        df_interpolated = df.interpolate(method='linear')
                        df.loc[month] = df_interpolated.loc[month]
                        continue
                except Exception as e:
                    print(f"    Erreur lors de l'interpolation pour {month}: {e}")

                # Stratégie 3: Utiliser la valeur du mois le plus proche
                print(f"    {month}: Utilisation du mois le plus proche")

                # Trouver le mois existant le plus proche
                month_date_ts = pd.Timestamp(month_date)
                existing_dates = [pd.Timestamp(datetime.strptime(x, '%Y-%m')) for x in existing_months]
                closest_idx = np.argmin([abs((date - month_date_ts).days) for date in existing_dates])
                closest_month = existing_months[closest_idx]

                df.loc[month] = df.loc[closest_month]

            # Mettre à jour les données avec les valeurs interpolées
            for month in all_months:
                # Convertir les valeurs en dictionnaire avec gestion des NaN
                values = df.loc[month].to_dict()
                clean_values = {k: float(v) for k, v in values.items() if not pd.isna(v)}

                if clean_values:  # Si des valeurs existent
                    data[country][month] = clean_values

    # Écrire le résultat en JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Traitement terminé. Fichier {output_json_file} créé/mis à jour.")
    return data


if __name__ == "__main__":
    # Fichier JSON existant
    input_file = "./data/air-quality/resultats_capteurs_par_pays.json"

    # Fichier de sortie (optionnel, si différent de l'entrée)
    output_file = "./data/air-quality/resultats_capteurs_par_pays_complet.json"

    # Exécuter le traitement
    fill_missing_months(input_file, output_file)
