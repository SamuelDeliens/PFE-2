import pandas as pd
import glob

def extract_locations_and_ids(df):
    """Extrait les localisations et leurs IDs à partir du dataframe."""
    location_ids = df.iloc[0].tolist()
    locations = df.iloc[1].tolist()

    location_mapping = {}
    for i in range(1, len(locations)):
        if pd.notna(location_ids[i]) and pd.notna(locations[i]):
            location_mapping[location_ids[i]] = locations[i]

    return location_mapping


def clean_air_quality_data(input_file, output_file):
    """Nettoie les données de qualité d'air et les sauvegarde."""
    df = pd.read_csv(input_file, header=None)

    location_mapping = extract_locations_and_ids(df)

    column_names = ['datetime'] + df.iloc[0, 1:].tolist()

    data_df = df.iloc[6:].copy()

    data_df.columns = column_names

    data_df['datetime'] = pd.to_datetime(data_df['datetime'])

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(output_file, index=False)

    return location_mapping


def process_all_files(base_dirs, output_dir):
    """Traite tous les fichiers dans les dossiers spécifiés."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_location_mappings = {}

    for base_dir in base_dirs:
        pollutant = os.path.basename(base_dir)

        pollutant_output_dir = os.path.join(output_dir, pollutant)
        if not os.path.exists(pollutant_output_dir):
            os.makedirs(pollutant_output_dir)

        csv_files = glob.glob(os.path.join(base_dir, "*.csv"))

        for input_file in csv_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(pollutant_output_dir, filename)

            print(f"Traitement de {input_file}...")

            location_mapping = clean_air_quality_data(input_file, output_file)

            all_location_mappings.update(location_mapping)

    locations_df = pd.DataFrame(list(all_location_mappings.items()), columns=['ID', 'Localisation'])
    locations_df.to_csv(os.path.join(output_dir, 'locations_mapping.csv'), index=False)

    print(f"Traitement terminé. Les fichiers nettoyés sont dans {output_dir}")
    print(f"Le mapping des localisations est sauvegardé dans {os.path.join(output_dir, 'locations_mapping.csv')}")


if __name__ == "__main__":
    import os

    input_dir = "./data/air-quality/idf/brut/"
    output_dir = "./data/air-quality/idf/clean/"

    base_dirs = [input_dir + folder for folder in os.listdir(input_dir)]
    process_all_files(base_dirs, output_dir)
