import os
import json
import pandas as pd


def process_csv_files_to_json(main_directory):
    """
    Traite tous les fichiers CSV dans une structure de dossiers par pays
    et génère un fichier JSON unique avec les moyennes par pays et par mois.

    Structure attendue:
    main_directory/
        pays1/
            capteur1.csv
            capteur2.csv
            ...
        pays2/
            capteur1.csv
            ...
    """
    result_data = {}

    fields_to_aggregate = ['value', 'min', 'q02', 'q25', 'median', 'q75', 'q98', 'max', 'avg', 'sd']

    for country in os.listdir(main_directory):
        country_path = os.path.join(main_directory, country)

        if not os.path.isdir(country_path):
            continue

        print(f"Traitement du pays: {country}")

        result_data[country] = {}

        country_data_by_date = {}

        for sensor_file in os.listdir(country_path):
            if not sensor_file.endswith('.csv'):
                continue

            file_path = os.path.join(country_path, sensor_file)

            try:
                df = pd.read_csv(file_path)

                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                for _, row in df.iterrows():
                    date = row['date']

                    if date not in country_data_by_date:
                        country_data_by_date[date] = []

                    sensor_values = {field: float(row[field]) for field in fields_to_aggregate}
                    country_data_by_date[date].append(sensor_values)

            except Exception as e:
                print(f"Erreur lors du traitement de {file_path}: {e}")

        for date, sensors_data in country_data_by_date.items():
            if not sensors_data:
                continue

            aggregated_values = {}
            for field in fields_to_aggregate:
                values = [sensor[field] for sensor in sensors_data]
                aggregated_values[field] = sum(values) / len(values)

            result_data[country][date] = aggregated_values

    output_file = 'resultats_capteurs_par_pays.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"Traitement terminé. Fichier {output_file} créé.")

    return result_data


if __name__ == "__main__":
    main_directory = "./data/air-quality/world"

    process_csv_files_to_json(main_directory)