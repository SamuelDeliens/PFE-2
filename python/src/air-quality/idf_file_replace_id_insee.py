import os
import pandas as pd


def replace_ids_with_insee(data_file, mapping_file, output_file):
    """
    Remplace les ID des capteurs par les codes INSEE correspondants dans un fichier de données.

    Args:
        data_file (str): Chemin vers le fichier de données à transformer
        mapping_file (str): Chemin vers le fichier de mapping contenant les codes INSEE
        output_file (str): Chemin où sauvegarder le fichier transformé
    """
    mapping_df = pd.read_csv(mapping_file)

    id_to_insee = dict(zip(mapping_df['ID'], mapping_df['INSEE_COM']))

    data_df = pd.read_csv(data_file)

    current_columns = data_df.columns[1:]

    rename_dict = {}
    for col in current_columns:
        if col in id_to_insee:
            rename_dict[col] = id_to_insee[col]
        else:
            print(f"Avertissement: ID {col} non trouvé dans le fichier de mapping")

    data_df = data_df.rename(columns=rename_dict)

    data_df.to_csv(output_file, index=False)
    print(f"Fichier transformé sauvegardé: {output_file}")


def process_all_data_files(data_dir, mapping_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)

        if rel_path != '.':
            output_subdir = os.path.join(output_dir, rel_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
        else:
            output_subdir = output_dir

        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_subdir, file)

                print(f"Traitement de {input_file}...")
                replace_ids_with_insee(input_file, mapping_file, output_file)


if __name__ == "__main__":
    data_dir = "./data/air-quality/idf/clean/"

    mapping_file = "./data/air-quality/idf/locations_mapping.csv"

    output_dir = "./data/air-quality/idf/final/"

    process_all_data_files(data_dir, mapping_file, output_dir)
    print(f"Tous les fichiers ont été traités. Les fichiers transformés sont dans {output_dir}")