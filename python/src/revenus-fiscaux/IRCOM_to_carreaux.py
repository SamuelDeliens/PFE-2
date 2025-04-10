import pandas as pd
import numpy as np
import os


def clean_ircom_data(fichier_ircom):
    """
    Nettoie les données IRCOM pour préparer le carroyage
    """
    ircom = pd.read_csv(fichier_ircom, sep=',', encoding='utf-8')

    ircom.columns = [
        'Dep', 'Commune', 'Libelle_commune', 'Tranche', 'Nb_foyers_fiscaux',
        'RFR_total', 'Impot_net', 'Nb_foyers_imposes', 'RFR_imposes',
        'Nb_foyers_salaires', 'Montant_salaires', 'Nb_foyers_retraites', 'Montant_retraites'
    ]

    ircom = ircom[ircom['Tranche'] == 'TOTAL']

    ircom['Dep'] = ircom['Dep'].astype(str)
    ircom['Commune'] = ircom['Commune'].astype(str)
    ircom['Code_INSEE'] = ircom['Dep'].str.zfill(2) + ircom['Commune'].str.zfill(3)

    colonnes_numeriques = [
        'Nb_foyers_fiscaux', 'RFR_total', 'Impot_net', 'Nb_foyers_imposes',
        'RFR_imposes', 'Nb_foyers_salaires', 'Montant_salaires',
        'Nb_foyers_retraites', 'Montant_retraites'
    ]

    for col in colonnes_numeriques:
        if ircom[col].dtype == 'object':
            ircom[col] = ircom[col].str.replace(' ', '')
            ircom[col] = ircom[col].str.replace(',', '.')
            ircom[col] = pd.to_numeric(ircom[col], errors='coerce')

    ircom['RFR_moyen_par_foyer'] = ircom['RFR_total'] / ircom['Nb_foyers_fiscaux']
    ircom['Impot_moyen_par_foyer_impose'] = ircom['Impot_net'] / ircom['Nb_foyers_imposes']
    ircom['Taux_imposition'] = ircom['Nb_foyers_imposes'] / ircom['Nb_foyers_fiscaux']

    print(f"Nombre de communes traitées: {len(ircom)}")
    return ircom


def prepare_for_carroyage(fichier_filosofi, ircom_cleaned):
    """
    Prépare les données pour la conversion en format carroyé
    en établissant la correspondance entre communes et carreaux
    """
    # Lire les données Filosofi existantes (200m)
    filosofi = pd.read_csv(fichier_filosofi, sep=',', encoding='utf-8')

    if 'Depcom' not in filosofi.columns:
        print("ERREUR: La colonne 'Depcom' n'est pas présente dans les données Filosofi")
        print("Impossible d'établir la correspondance commune-carreau")
        return None

    correspondance = filosofi[['Idcar_200m', 'Depcom']].drop_duplicates()

    poids = filosofi.groupby(['Depcom', 'Idcar_200m'])['Men'].sum().reset_index()
    total_menages = poids.groupby('Depcom')['Men'].sum().reset_index()
    total_menages.rename(columns={'Men': 'Total_Men'}, inplace=True)

    poids = pd.merge(poids, total_menages, on='Depcom')
    poids['Poids'] = poids['Men'] / poids['Total_Men']

    print(f"Table de correspondance créée pour {len(correspondance)} carreaux")
    return poids


def ventiler_donnees(ircom_cleaned, table_poids):
    """
    Ventile les données IRCOM aux carreaux selon les poids calculés
    """
    ventilation = pd.merge(
        table_poids,
        ircom_cleaned,
        left_on='Depcom',
        right_on='Code_INSEE',
        how='left'
    )

    variables_a_ventiler = [
        'Nb_foyers_fiscaux', 'RFR_total', 'Impot_net',
        'Nb_foyers_imposes', 'RFR_imposes'
    ]

    for var in variables_a_ventiler:
        ventilation[f'{var}_carreau'] = ventilation[var] * ventilation['Poids']

    resultat = ventilation[[
        'Idcar_200m', 'Depcom', 'Poids',
        'Nb_foyers_fiscaux_carreau', 'RFR_total_carreau',
        'Impot_net_carreau', 'Nb_foyers_imposes_carreau'
    ]]

    print(f"Données ventilées sur {len(resultat)} carreaux")
    return resultat


def creer_format_filosofi(donnees_ventilees, fichier_filosofi_modele):
    """
    Crée un fichier au format Filosofi avec les données IRCOM ventilées
    """
    filosofi_modele = pd.read_csv(fichier_filosofi_modele, sep=',', encoding='utf-8')

    resultat = pd.merge(
        filosofi_modele,
        donnees_ventilees,
        on='Idcar_200m',
        how='left'
    )

    resultat['IRCOM_Nb_foyers'] = resultat['Nb_foyers_fiscaux_carreau']
    resultat['IRCOM_RFR'] = resultat['RFR_total_carreau']
    resultat['IRCOM_Impot'] = resultat['Impot_net_carreau']

    print(f"Fichier au format Filosofi créé avec {len(resultat)} carreaux")
    return resultat


def transformer_ircom_to_filosofi(fichier_ircom, fichier_filosofi, annee):
    """
    Transforme les données IRCOM en format carroyé compatible avec Filosofi
    """
    print(f"Traitement des données IRCOM {annee}...")

    ircom_cleaned = clean_ircom_data(fichier_ircom)

    table_poids = prepare_for_carroyage(fichier_filosofi, ircom_cleaned)
    if table_poids is None:
        return

    donnees_ventilees = ventiler_donnees(ircom_cleaned, table_poids)

    resultat = creer_format_filosofi(donnees_ventilees, fichier_filosofi)

    nom_fichier_sortie = f"IRCOM_{annee}_format_Filosofi.csv"
    resultat.to_csv(nom_fichier_sortie, index=False)
    print(f"Fichier sauvegardé: {nom_fichier_sortie}")


def creer_serie_temporelle(repertoire_ircom, fichier_filosofi_modele, annees):
    """
    Crée une série temporelle complète des données IRCOM au format carroyé
    pour toutes les années demandées
    """
    resultats_par_annee = {}

    for annee in annees:
        fichier_ircom = os.path.join(repertoire_ircom, f"IRCOM_{annee}.csv")

        if not os.path.exists(fichier_ircom):
            print(f"Attention: Le fichier {fichier_ircom} n'existe pas, année {annee} ignorée")
            continue

        print(f"\nTraitement de l'année {annee}...")
        ircom_cleaned = clean_ircom_data(fichier_ircom)
        table_poids = prepare_for_carroyage(fichier_filosofi_modele, ircom_cleaned)

        if table_poids is not None:
            donnees_ventilees = ventiler_donnees(ircom_cleaned, table_poids)
            resultats_par_annee[annee] = donnees_ventilees

    if len(resultats_par_annee) > 1:
        print("\nInterpolation des années manquantes...")
        annees_disponibles = sorted(list(resultats_par_annee.keys()))
        toutes_annees = list(range(min(annees_disponibles), max(annees_disponibles) + 1))

        annees_manquantes = [a for a in toutes_annees if a not in annees_disponibles]

        if annees_manquantes:
            carreaux = resultats_par_annee[annees_disponibles[0]]['Idcar_200m'].unique()

            for carreau in carreaux:
                series = {}
                for annee in annees_disponibles:
                    carreau_data = resultats_par_annee[annee][
                        resultats_par_annee[annee]['Idcar_200m'] == carreau
                        ]
                    if not carreau_data.empty:
                        series[annee] = carreau_data.iloc[0]

                for annee_manquante in annees_manquantes:
                    annees_avant = [a for a in annees_disponibles if a < annee_manquante]
                    annees_apres = [a for a in annees_disponibles if a > annee_manquante]

                    if annees_avant and annees_apres:
                        annee_avant = max(annees_avant)
                        annee_apres = min(annees_apres)

                        poids_avant = (annee_apres - annee_manquante) / (annee_apres - annee_avant)
                        poids_apres = (annee_manquante - annee_avant) / (annee_apres - annee_avant)

                        if annee_avant in series and annee_apres in series:
                            interp = series[annee_avant].copy()

                            cols_num = [
                                'Nb_foyers_fiscaux_carreau', 'RFR_total_carreau',
                                'Impot_net_carreau', 'Nb_foyers_imposes_carreau'
                            ]

                            for col in cols_num:
                                interp[col] = (
                                        series[annee_avant][col] * poids_avant +
                                        series[annee_apres][col] * poids_apres
                                )

                            if annee_manquante not in resultats_par_annee:
                                resultats_par_annee[annee_manquante] = pd.DataFrame(columns=series[annee_avant].index)

                            resultats_par_annee[annee_manquante] = resultats_par_annee[annee_manquante].append(
                                interp, ignore_index=True
                            )

    for annee, resultat in resultats_par_annee.items():
        nom_fichier = f"IRCOM_{annee}_format_Filosofi.csv"
        resultat.to_csv(nom_fichier, index=False)
        print(f"Fichier sauvegardé: {nom_fichier}")


if __name__ == "__main__":
    # Ces variables seraient à remplacer par vos propres chemins de fichiers
    #fichier_ircom = "IRCOM_2015.csv"
    #fichier_filosofi = "Filosofi_carreaux_200m_2015.csv"
    #annee = "2015"

    #transformer_ircom_to_filosofi(fichier_ircom, fichier_filosofi, annee)

    repertoire_ircom = "./data/revenus-fiscaux/IRCOM/csv/"
    fichier_filosofi_modele = "./data/revenus-fiscaux/Filosofi/Filosofi2019_carreaux_200m_csv/carreaux_200m_met.csv"
    annees = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

    creer_serie_temporelle(repertoire_ircom, fichier_filosofi_modele, annees)