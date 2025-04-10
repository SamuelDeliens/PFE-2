import pandas as pd

base_dir = "./data/"
ircom_dir = base_dir + "revenus-fiscaux/IRCOM/"

def safe_float(value):
    try:
        # Nettoyer la valeur (supprimer espaces, remplacer virgules par points)
        if isinstance(value, str):
            value = value.strip().replace(',', '.')
            # Si la valeur est juste un point ou vide après nettoyage
            if value in ['.', '', ' ', '. ']:
                return 0.0
        return float(value)
    except (ValueError, TypeError):
        # En cas d'erreur, retourner 0
        return 0.0

# Pour chaque année (2015, 2017, 2019)
for annee in [2015, 2017, 2019]:
    # Charger la table de correspondance pour cette année
    correspondance = pd.read_csv(f"{base_dir}correspondence_table_carreaux_communes_{annee}.csv")

    # Charger les données IRCOM pour cette année
    ircom_data = pd.read_csv(f"{ircom_dir}clean/IRCOM_{annee}.csv")

    # Stocker les résultats dans une liste
    resultats_list = []

    # Grouper par ID de carreau
    for id_carreau, groupe in correspondance.groupby('Idcar_200m'):
        # Variables à calculer pour ce carreau
        revenu_fiscal = 0
        nb_foyers = 0
        nb_foyers_imposes = 0
        revenu_fiscal_imposes = 0
        traitements_salaires = 0
        retraites_pensions = 0

        # Pour chaque commune qui chevauche ce carreau
        for _, row in groupe.iterrows():
            code_commune = row['INSEE_COM']
            poids = row['weight']

            # Récupérer les données IRCOM pour cette commune
            ircom_commune_df = ircom_data[ircom_data['INSEE_COM'] == code_commune]

            # Vérifier si des données existent pour cette commune
            if ircom_commune_df.empty:
                continue

            ircom_commune = ircom_commune_df.iloc[0]

            # Ventiler les variables selon le poids
            revenu_fiscal += safe_float(ircom_commune['Revenu fiscal de référence des foyers fiscaux']) * poids
            nb_foyers += safe_float(ircom_commune['Nombre de foyers fiscaux']) * poids
            nb_foyers_imposes += safe_float(ircom_commune['Nombre de foyers fiscaux imposés']) * poids
            revenu_fiscal_imposes += safe_float(
                ircom_commune['Revenu fiscal de référence des foyers fiscaux imposés']) * poids
            traitements_salaires += safe_float(ircom_commune['Montant']) * poids  # Traitements et salaires
            retraites_pensions += safe_float(ircom_commune['Montant.1']) * poids  # Retraites et pensions

        # Ajouter ce carreau au résultat
        resultats_list.append({
            'Idcar_200m': id_carreau,
            'revenu_fiscal': revenu_fiscal,
            'nb_foyers': nb_foyers,
            'nb_foyers_imposes': nb_foyers_imposes,
            'revenu_fiscal_imposes': revenu_fiscal_imposes,
            'traitements_salaires': traitements_salaires,
            'retraites_pensions': retraites_pensions,
            'annee': annee
        })

    # Créer le DataFrame à partir de la liste
    carreaux_resultat = pd.DataFrame(resultats_list)

    # Sauvegarder les résultats pour cette année
    carreaux_resultat.to_csv(f"{ircom_dir}carroye_200m/ircom_carroye_200m_{annee}.csv", index=False)