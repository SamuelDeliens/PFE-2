import os
import pandas as pd

def clean_dep(dep):
    return str(dep).rstrip('0')


def generate_INSEE_code(dep, commune):
    cleaned_dep = clean_dep(dep)
    formated_dep = str(cleaned_dep).zfill(2)
    com_formate = str(commune).zfill(3)
    return f"{formated_dep}{com_formate}"


def compute_csv_file(input, output):
    os.makedirs(output, exist_ok=True)

    csv_file = [f for f in os.listdir(input) if f.endswith('.csv')]

    exclude_dep = ['971', '972', '973', '974', '976', 'B31']

    for file in csv_file:
        path_input = os.path.join(input, file)
        path_output = os.path.join(output, file)

        try:
            df = pd.read_csv(
                path_input,
                sep=',',
                encoding='utf-8',
                header=0,
                skipinitialspace=True,
                dtype=str
            )

            df.columns = [col.strip() for col in df.columns]

            needed_cols = ['Libellé de la commune', 'Revenu fiscal de référence par tranche (en euros)']
            if all(col in df.columns for col in needed_cols):
                clean_df = df[~df['Dép.'].isin(exclude_dep)]

                clean_df = clean_df[
                    clean_df['Revenu fiscal de référence par tranche (en euros)'].str.strip().str.upper() == 'TOTAL'
                ]

                # Modifier la colonne du département
                clean_df['Dép.'] = clean_df['Dép.'].apply(clean_dep)

                # Ajoute la colonne INSEE_COM
                clean_df['INSEE_COM'] = clean_df.apply(
                    lambda row: generate_INSEE_code(row['Dép.'], row['Commune']),
                    axis=1
                )

                # Supprime les colonnes inutiles
                colonnes_a_supprimer = ['Revenu fiscal de référence par tranche (en euros)', 'Commune', 'Dép.']
                clean_df = clean_df.drop(columns=colonnes_a_supprimer, errors='ignore')

                # Move INSEE_COM to the front
                colonnes = ['INSEE_COM'] + [col for col in clean_df.columns if col != 'INSEE_COM']
                clean_df = clean_df[colonnes]

                # Sauvegarde le fichier modifié
                clean_df.to_csv(path_output, sep=',', encoding='utf-8', index=False)

                print(f"Traitement terminé pour {file}")
                print(f"Nombre total de lignes : {len(df)}")
                print(f"Nombre de lignes après filtrage : {len(clean_df)}")
                print(f"Lignes supprimées : {len(df) - len(clean_df)}")
                print("-" * 50)
            else:
                print(f"Colonnes manquantes dans {file}. Colonnes présentes : {df.columns}")

        except Exception as e:
            print(f"Erreur lors du traitement de {file}: {e}")


if __name__ == "__main__":
    output_dir = "./data/revenus-fiscaux/IRCOM/csv"
    input_dir = "./data/revenus-fiscaux/IRCOM/clean"

    compute_csv_file(output_dir, input_dir)