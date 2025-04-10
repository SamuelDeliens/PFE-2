import pandas as pd
import os
import numpy as np


def harmonize_filosofi_data(input_files, output_dir, id_col):
    """
    Harmonise les données Filosofi en standardisant les noms de colonnes,
    en ajoutant les colonnes manquantes et en réorganisant l'ordre des colonnes.

    Args:
        input_files (list): Liste des chemins vers les fichiers CSV/Excel à harmoniser
        output_dir (str): Dossier où sauvegarder les fichiers harmonisés
    """
    os.makedirs(output_dir, exist_ok=True)

    common_columns_mapping = {
        'ind': 'Ind',
        'men': 'Men',
        'men_pauv': 'Men_pauv',
        'men_1ind': 'Men_1ind',
        'men_5ind': 'Men_5ind',
        'men_prop': 'Men_prop',
        'men_fmp': 'Men_fmp',
        'ind_snv': 'Ind_snv',
        'men_surf': 'Men_surf',
        'men_coll': 'Men_coll',
        'men_mais': 'Men_mais',
        'log_av45': 'Log_av45',
        'log_45_70': 'Log_45_70',
        'log_70_90': 'Log_70_90',
        'log_ap90': 'Log_ap90',
        'log_inc': 'Log_inc',
        'log_soc': 'Log_soc',
        'ind_0_3': 'Ind_0_3',
        'ind_4_5': 'Ind_4_5',
        'ind_6_10': 'Ind_6_10',
        'ind_11_17': 'Ind_11_17',
        'ind_18_24': 'Ind_18_24',
        'ind_25_39': 'Ind_25_39',
        'ind_40_54': 'Ind_40_54',
        'ind_55_64': 'Ind_55_64',
        'ind_65_79': 'Ind_65_79',
        'ind_80p': 'Ind_80p',
        'ind_inc': 'Ind_inc',

        # Standardisation des identifiants spatiaux
        'IdINSPIRE': 'Idcar_200m',
        'Id_carr1km': 'Idcar_1km',
        'Id_carr_n': 'Idcar_nat',
        'I_est_cr': 'I_est_200',
        'idcar_200m': 'Idcar_200m',
        'idcar_1km': 'Idcar_1km',
        'idcar_nat': 'Idcar_nat',
        'i_est_200': 'I_est_200',
        'i_est_1km': 'I_est_1km'
    }

    additional_columns = ['Depcom', 'lcog_geo']#, 'Id_car2010']

    standard_column_order = [
        'Idcar_200m', 'Idcar_1km', 'Idcar_nat', 'Depcom', 'lcog_geo', #'Id_car2010',

        'Ind', 'Men', 'Men_pauv', 'Men_1ind', 'Men_5ind', 'Men_prop', 'Men_fmp',
        'Ind_snv', 'Men_surf', 'Men_coll', 'Men_mais',

        'Log_av45', 'Log_45_70', 'Log_70_90', 'Log_ap90', 'Log_inc', 'Log_soc',

        'Ind_0_3', 'Ind_4_5', 'Ind_6_10', 'Ind_11_17', 'Ind_18_24',
        'Ind_25_39', 'Ind_40_54', 'Ind_55_64', 'Ind_65_79', 'Ind_80p', 'Ind_inc',

        'I_est_200', 'I_est_1km'
    ]

    dataframes = []
    reference_data = {}

    for input_file in input_files:
        file_ext = os.path.splitext(input_file)[1].lower()

        if file_ext == '.csv':
            try:
                df = pd.read_csv(input_file, sep=',', encoding='utf-8')
            except:
                df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        else:
            print(f"Format de fichier non pris en charge: {file_ext}")
            continue

        df = df.rename(columns=common_columns_mapping)

        dataframes.append((input_file, df))

        if id_col in df.columns:
            for col in additional_columns:
                if col in df.columns and not df[col].isna().all():
                    if col not in reference_data:
                        reference_data[col] = {}

                    current_ref = dict(zip(df[id_col], df[col]))
                    reference_data[col].update(current_ref)

    for input_file, df in dataframes:
        file_ext = os.path.splitext(input_file)[1].lower()

        if 'I_pauv' in df.columns:
            df = df.drop(columns=['I_pauv'])

        for col in additional_columns:
            if col not in df.columns:
                df[col] = np.nan

            if col in reference_data:
                for idx, row in df.iterrows():
                    if pd.isna(df.at[idx, col]) and id_col in df.columns:
                        idcar_value = row[id_col]
                        if idcar_value in reference_data[col]:
                            df.at[idx, col] = reference_data[col][idcar_value]

        existing_columns = [col for col in standard_column_order if col in df.columns]

        for col in df.columns:
            if col not in existing_columns:
                existing_columns.append(col)

        df = df[existing_columns]

        base_filename = os.path.basename(input_file)
        output_filename = f"harmonized_{base_filename}"
        output_path = os.path.join(output_dir, output_filename)

        if file_ext == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif file_ext in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)

        print(f"Fichier harmonisé créé: {output_path}")


if __name__ == "__main__":
    base_path = "./data/revenus-fiscaux/Filosofi/brut/"
    input_files = [
        base_path + "Filosofi2015_carreaux_1000m_csv/Filosofi2015_carreaux_1000m_metropole.csv",
        base_path + "Filosofi2017_carreaux_1km_csv/Filosofi2017_carreaux_1km_met.csv",
        base_path + "Filosofi2019_carreaux_1km_csv/carreaux_1km_met.csv",
    ]
    output_dir = base_path + "harmonized/"

    id_col = 'Idcar_1km'

    harmonize_filosofi_data(input_files, output_dir, id_col)