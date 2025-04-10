import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import geopandas as gpd
import sys
import argparse
from pathlib import Path


# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---

def charger_donnees_pollution(dossier, annee):
    """
    Charge les données de pollution pour un dossier spécifique (NO2, NOx, PM10, PM25)
    """
    print(f"Chargement des données de {dossier} pour {annee}...")

    # Liste tous les fichiers correspondant au pattern
    fichiers = glob.glob(os.path.join(dossier, f"{annee}.csv"))

    if not fichiers:
        print(f"ATTENTION: Aucun fichier trouvé pour {dossier} et {annee}")
        return pd.DataFrame()

    # Dataframe pour stocker toutes les données
    donnees_completes = pd.DataFrame()

    # Analyser le premier fichier pour comprendre sa structure
    try:
        # Lire quelques lignes pour déterminer la structure
        with open(fichiers[0], 'r') as f:
            premiere_ligne = f.readline().strip()
            deuxieme_ligne = f.readline().strip() if f.readline() else ""

        print(f"Analyse de la structure du fichier: {fichiers[0]}")
        print(f"Première ligne: {premiere_ligne[:100]}...")
        print(f"Deuxième ligne: {deuxieme_ligne[:100]}...")

        # Déterminer si le format est celui attendu
        colonnes = premiere_ligne.split(',')
        if colonnes and colonnes[0] == "datetime":
            print("Format reconnu: première colonne = 'datetime'")
        else:
            print("Format atypique, tentative d'adaptation...")

    except Exception as e:
        print(f"Erreur lors de l'analyse préliminaire: {e}")

    # Charger les fichiers
    for fichier in fichiers:
        try:
            # Lire le fichier en essayant de détecter automatiquement le format
            df = pd.read_csv(fichier)

            # Vérifier si 'datetime' est une colonne
            if 'datetime' in df.columns:
                # Forcer la conversion en datetime
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                # Définir comme index
                df.set_index('datetime', inplace=True)
            else:
                # Essayer de trouver une colonne de date/heure
                for col in df.columns:
                    # Vérifier si la colonne semble contenir des dates
                    if df[col].dtype == 'object' and df[col].iloc[0] and (
                            '-' in str(df[col].iloc[0]) or '/' in str(df[col].iloc[0])):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            df.set_index(col, inplace=True)
                            print(f"Colonne {col} utilisée comme index de date/heure")
                            break
                        except:
                            continue

            # Fusion avec les données existantes
            if donnees_completes.empty:
                donnees_completes = df
            else:
                donnees_completes = pd.concat([donnees_completes, df])

        except Exception as e:
            print(f"Erreur lors du chargement du fichier {fichier}: {e}")

    # Vérifier que l'index est bien de type DatetimeIndex
    if not isinstance(donnees_completes.index, pd.DatetimeIndex):
        print("ATTENTION: L'index n'est pas un DatetimeIndex. Tentative de conversion forcée...")

        # Si les données sont chargées mais l'index n'est pas DatetimeIndex, forcer la conversion
        try:
            # Réinitialiser l'index et recréer un DataFrame
            df_reset = donnees_completes.reset_index()

            # Identifier la colonne qui pourrait contenir des dates
            if 'datetime' in df_reset.columns or 'index' in df_reset.columns:
                date_col = 'datetime' if 'datetime' in df_reset.columns else 'index'

                # Vérifier le type de la colonne
                if df_reset[date_col].dtype == 'object':
                    # Essayer de convertir la colonne en datetime
                    df_reset[date_col] = pd.to_datetime(df_reset[date_col], errors='coerce')
                    donnees_completes = df_reset.set_index(date_col)
                    print(f"Conversion forcée de l'index en DatetimeIndex via colonne {date_col}")
                elif pd.api.types.is_datetime64_any_dtype(df_reset[date_col]):
                    # La colonne est déjà en datetime, définir comme index
                    donnees_completes = df_reset.set_index(date_col)
                    print(f"Redéfinition de l'index en utilisant colonne {date_col} déjà en datetime")

            # Vérifier à nouveau
            if not isinstance(donnees_completes.index, pd.DatetimeIndex):
                print("ÉCHEC: La conversion en DatetimeIndex a échoué.")
        except Exception as e:
            print(f"Erreur lors de la conversion forcée de l'index: {e}")
    else:
        print("Index correctement identifié comme DatetimeIndex.")

    # Afficher des informations sur les données chargées
    if not donnees_completes.empty:
        print(f"Données chargées: {donnees_completes.shape[0]} lignes, {donnees_completes.shape[1]} colonnes")
        print(f"Période: de {donnees_completes.index.min()} à {donnees_completes.index.max()}")
    else:
        print("Aucune donnée n'a pu être chargée.")

    return donnees_completes


def charger_donnees_fiscales(file):
    """
    Charge les données fiscales IRCOM

    Parameters:
    -----------
    chemin_dossier : str
        Chemin du dossier contenant les données fiscales (peut être un dossier ou un fichier spécifique)
    """
    print(f"Chargement des données fiscales depuis {file}...")

    if not os.path.exists(file):
        file += ".csv"  # Essayer d'ajouter l'extension si nécessaire
        if not os.path.exists(file):
            print(f"Fichier non trouvé: {file}")
            return None

    # Chargement des données
    try:
        print(f"Chargement du fichier: {file}")
        # Analyser les premières lignes pour diagnostiquer le format
        with open(file, 'r', encoding='utf-8') as f:
            premieres_lignes = [f.readline() for _ in range(3)]

        print("Aperçu des premières lignes:")
        for i, ligne in enumerate(premieres_lignes):
            print(f"Ligne {i + 1}: {ligne[:100]}...")

        # Charger le CSV avec détection des types de données
        donnees_fiscales = pd.read_csv(file, sep=',', encoding='utf-8')

        print(f"Colonnes détectées: {donnees_fiscales.columns.tolist()}")

        # Vérifier les types de données
        print("Types de données détectés:")
        for col in donnees_fiscales.columns:
            print(f"- {col}: {donnees_fiscales[col].dtype}")

        # Vérifier si les colonnes attendues sont présentes
        colonnes_requises = ['INSEE_COM', 'Nombre de foyers fiscaux', 'Revenu fiscal de référence des foyers fiscaux',
                             'Nombre de foyers fiscaux imposés']

        # Vérifier si toutes les colonnes requises sont présentes
        if not all(col in donnees_fiscales.columns for col in colonnes_requises):
            print("AVERTISSEMENT: Certaines colonnes requises sont manquantes dans les données fiscales")
            print(f"Colonnes trouvées: {donnees_fiscales.columns.tolist()}")

            # Essayer de faire correspondre les noms de colonnes proches
            correspondances = {
                'INSEE_COM': ['code_insee', 'code commune', 'code_commune', 'codecommune'],
                'Nombre de foyers fiscaux': ['nb_foyers', 'foyers_fiscaux', 'nombre_foyers'],
                'Revenu fiscal de référence des foyers fiscaux': ['revenu_fiscal_reference', 'rfr', 'revenu_reference'],
                'Nombre de foyers fiscaux imposés': ['nb_foyers_imposes', 'foyers_imposes', 'nombre_imposes']
            }

            # Renommer les colonnes si possible
            for col_req, alternatives in correspondances.items():
                if col_req not in donnees_fiscales.columns:
                    for alt in alternatives:
                        if alt in donnees_fiscales.columns:
                            donnees_fiscales.rename(columns={alt: col_req}, inplace=True)
                            print(f"Colonne '{alt}' renommée en '{col_req}'")
                            break

        # Vérifier à nouveau après renommage
        if not all(col in donnees_fiscales.columns for col in colonnes_requises):
            print("ERREUR: Impossible de trouver toutes les colonnes requises même après renommage")
            return None

        # S'assurer que le code INSEE est en format string
        donnees_fiscales['INSEE_COM'] = donnees_fiscales['INSEE_COM'].astype(str)

        # Nettoyer les codes INSEE (enlever les espaces et autres caractères non numériques)
        donnees_fiscales['INSEE_COM'] = donnees_fiscales['INSEE_COM'].str.strip().str.replace(r'\D', '', regex=True)

        # Convertir les colonnes numériques en float, en gérant les formats potentiellement problématiques
        for col in ['Nombre de foyers fiscaux', 'Revenu fiscal de référence des foyers fiscaux',
                    'Nombre de foyers fiscaux imposés']:
            # Vérifier si la colonne contient des chaînes de caractères
            if donnees_fiscales[col].dtype == 'object':
                print(f"Conversion de la colonne {col} en numérique...")
                # Remplacer les virgules par des points et convertir en float
                donnees_fiscales[col] = donnees_fiscales[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                donnees_fiscales[col] = pd.to_numeric(donnees_fiscales[col], errors='coerce')

        # Calculer des indicateurs économiques supplémentaires
        print("Calcul des indicateurs économiques...")
        donnees_fiscales['Revenu_moyen_par_foyer'] = donnees_fiscales['Revenu fiscal de référence des foyers fiscaux'] / \
                                                     donnees_fiscales['Nombre de foyers fiscaux'].replace(0, np.nan)
        donnees_fiscales['Pourcentage_foyers_imposes'] = (donnees_fiscales['Nombre de foyers fiscaux imposés'] /
                                                          donnees_fiscales['Nombre de foyers fiscaux'].replace(0,
                                                                                                               np.nan)) * 100

        # Vérifier s'il y a des valeurs non numériques dans les colonnes calculées
        for col in ['Revenu_moyen_par_foyer', 'Pourcentage_foyers_imposes']:
            if donnees_fiscales[col].isnull().any():
                print(
                    f"AVERTISSEMENT: La colonne {col} contient {donnees_fiscales[col].isnull().sum()} valeurs manquantes")

        # Afficher quelques statistiques
        print(f"Données fiscales chargées: {donnees_fiscales.shape[0]} communes")
        if not donnees_fiscales['Revenu_moyen_par_foyer'].isnull().all():
            print(
                f"Revenu moyen min: {donnees_fiscales['Revenu_moyen_par_foyer'].min():.2f}, max: {donnees_fiscales['Revenu_moyen_par_foyer'].max():.2f}")

        return donnees_fiscales

    except Exception as e:
        print(f"Erreur lors du chargement des données fiscales: {e}")
        # Afficher plus de détails sur l'erreur
        import traceback
        traceback.print_exc()
        return None


def pretraiter_donnees_pollution(df_pollution, polluant):
    """
    Prétraitement des données de pollution
    """
    print(f"Prétraitement des données de pollution pour {polluant}...")

    # Vérifier si le DataFrame est vide
    if df_pollution.empty:
        print(f"ERREUR: DataFrame de pollution {polluant} vide")
        # Retourner des DataFrames vides pour éviter l'erreur
        return pd.DataFrame(
            columns=['INSEE_COM', f'Moyenne_{polluant}']), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Afficher des informations sur le DataFrame initial
    print(f"Données initiales: {df_pollution.shape[0]} lignes, {df_pollution.shape[1]} colonnes")
    print(f"Type d'index: {type(df_pollution.index)}")

    # Vérifier que l'index est bien de type DatetimeIndex
    if not isinstance(df_pollution.index, pd.DatetimeIndex):
        print("ATTENTION: L'index n'est pas un DatetimeIndex, tentative alternative de traitement...")

        # Méthode alternative sans resample: calculer directement les moyennes par colonne
        moyennes_par_colonne = df_pollution.mean(skipna=True)

        # Créer un DataFrame avec les moyennes
        df_moyennes = pd.DataFrame({
            'INSEE_COM': moyennes_par_colonne.index,
            f'Moyenne_{polluant}': moyennes_par_colonne.values
        })

        # Convertir les codes INSEE en strings et nettoyer
        df_moyennes['INSEE_COM'] = df_moyennes['INSEE_COM'].astype(str)

        # Supprimer les lignes avec des codes INSEE non valides
        # Par exemple, si le code INSEE est 'datetime' ou autre chose de non numérique
        df_moyennes = df_moyennes[df_moyennes['INSEE_COM'].str.match(r'\d+')]

        print(f"Traitement alternatif terminé: {df_moyennes.shape[0]} communes avec des moyennes calculées")

        # Dans ce cas, on ne peut pas calculer les moyennes temporelles
        return df_moyennes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Si l'index est bien un DatetimeIndex, procéder comme prévu
    print("Index DatetimeIndex confirmé, calcul des statistiques temporelles...")

    # Convertir les colonnes en nombres et gérer les valeurs manquantes
    for col in df_pollution.columns:
        df_pollution[col] = pd.to_numeric(df_pollution[col], errors='coerce')

    try:
        # Créer des moyennes temporelles
        moyennes_journalieres = df_pollution.resample('D').mean()
        moyennes_mensuelles = df_pollution.resample('M').mean()
        moyennes_annuelles = df_pollution.resample('Y').mean()

        print(
            f"Moyennes calculées: journalières ({moyennes_journalieres.shape[0]} jours), mensuelles ({moyennes_mensuelles.shape[0]} mois), annuelles ({moyennes_annuelles.shape[0]} années)")

        # Calculer la moyenne pour chaque colonne (code INSEE)
        moyennes_par_commune = df_pollution.mean(skipna=True)

        # Créer un dataframe avec les codes INSEE et moyennes
        df_moyennes = pd.DataFrame({
            'INSEE_COM': moyennes_par_commune.index,
            f'Moyenne_{polluant}': moyennes_par_commune.values
        })

        # Convertir les codes INSEE en strings pour la jointure ultérieure
        df_moyennes['INSEE_COM'] = df_moyennes['INSEE_COM'].astype(str)

        # Filtrer les valeurs non numériques ou invalides dans les codes INSEE
        df_moyennes = df_moyennes[df_moyennes['INSEE_COM'].str.match(r'\d+')]

        print(f"Moyennes par commune calculées: {df_moyennes.shape[0]} communes")

        return df_moyennes, moyennes_journalieres, moyennes_mensuelles, moyennes_annuelles

    except Exception as e:
        print(f"ERREUR lors du resampling ou traitement: {e}")
        print("Tentative de méthode alternative...")

        # Méthode alternative en cas d'erreur
        try:
            moyennes_par_colonne = df_pollution.mean(skipna=True)

            df_moyennes = pd.DataFrame({
                'INSEE_COM': moyennes_par_colonne.index,
                f'Moyenne_{polluant}': moyennes_par_colonne.values
            })

            df_moyennes['INSEE_COM'] = df_moyennes['INSEE_COM'].astype(str)
            df_moyennes = df_moyennes[df_moyennes['INSEE_COM'].str.match(r'\d+')]

            print(f"Méthode alternative réussie: {df_moyennes.shape[0]} communes")
            return df_moyennes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        except Exception as e2:
            print(f"ERREUR avec la méthode alternative: {e2}")
            return pd.DataFrame(
                columns=['INSEE_COM', f'Moyenne_{polluant}']), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def fusionner_donnees(dfs_pollution, df_fiscales):
    """
    Fusionne les différentes données de pollution et les données fiscales

    Parameters:
    -----------
    dfs_pollution : dict
        Dictionnaire contenant les DataFrames prétraités pour chaque polluant
    df_fiscales : DataFrame
        DataFrame contenant les données fiscales

    Returns:
    --------
    DataFrame
        DataFrame fusionné contenant toutes les données
    """
    print("Fusion des données...")

    if not dfs_pollution:
        print("ERREUR: Aucune donnée de pollution disponible pour la fusion")
        return pd.DataFrame()

    if df_fiscales is None or df_fiscales.empty:
        print("ERREUR: Données fiscales non disponibles pour la fusion")
        return pd.DataFrame()

    # Initialiser le DataFrame fusionné avec le premier DataFrame de pollution
    premier_polluant = list(dfs_pollution.keys())[0]
    print(f"Initialisation avec {premier_polluant}")
    df_pollution = dfs_pollution[premier_polluant].copy()

    # Fusionner avec les autres DataFrames de pollution
    for polluant, df in dfs_pollution.items():
        if polluant != premier_polluant:
            print(f"Fusion avec {polluant}")
            df_pollution = df_pollution.merge(df, on='INSEE_COM', how='outer')

    # Créer un indice composite de pollution
    colonnes_pollution = [col for col in df_pollution.columns if col.startswith('Moyenne_')]
    if colonnes_pollution:
        print(f"Calcul de l'indice composite basé sur: {colonnes_pollution}")
        df_pollution['Indice_pollution_composite'] = df_pollution[colonnes_pollution].mean(axis=1, skipna=True)
    else:
        print("AVERTISSEMENT: Aucune colonne de pollution trouvée pour calculer l'indice composite")
        return pd.DataFrame()

    # Fusionner avec les données fiscales
    print("Fusion avec les données fiscales")
    df_complet = df_pollution.merge(df_fiscales, on='INSEE_COM', how='inner')

    print(f"Fusion terminée: {df_complet.shape[0]} communes avec données complètes")

    return df_complet


# --- 2. ANALYSE EXPLORATOIRE ---

def analyse_exploratoire(df_complet):
    """
    Réalise une analyse exploratoire des données
    """
    print("Analyse exploratoire des données...")

    # Vérifier si le DataFrame est vide ou trop petit
    if df_complet.empty or len(df_complet) < 5:
        print("ERREUR: Données insuffisantes pour l'analyse exploratoire")
        return None, None, None

    # Statistiques descriptives
    try:
        stats_desc = df_complet.describe()
        print("Statistiques descriptives calculées")
    except Exception as e:
        print(f"ERREUR lors du calcul des statistiques descriptives: {e}")
        stats_desc = None

    # Calcul des corrélations
    try:
        # Identifier les colonnes pour l'analyse
        colonnes_pollution = [col for col in df_complet.columns if col.startswith('Moyenne_')]
        if 'Indice_pollution_composite' in df_complet.columns:
            colonnes_pollution.append('Indice_pollution_composite')

        colonnes_economiques = []
        for col in ['Revenu_moyen_par_foyer', 'Pourcentage_foyers_imposes']:
            if col in df_complet.columns:
                colonnes_economiques.append(col)

        # Calculer les corrélations si nous avons les colonnes nécessaires
        if colonnes_pollution and colonnes_economiques:
            colonnes_analyse = colonnes_pollution + colonnes_economiques
            correlations = df_complet[colonnes_analyse].corr()
            print("Matrice de corrélation calculée")

            # Afficher les principales corrélations
            for col_pol in colonnes_pollution:
                for col_eco in colonnes_economiques:
                    if col_pol in correlations.index and col_eco in correlations.columns:
                        corr = correlations.loc[col_pol, col_eco]
                        print(f"Corrélation {col_pol} vs {col_eco}: {corr:.4f}")
        else:
            print("AVERTISSEMENT: Pas assez de colonnes pour calculer les corrélations")
            correlations = None
    except Exception as e:
        print(f"ERREUR lors du calcul des corrélations: {e}")
        correlations = None

    # Division en quartiles de revenus et calcul de la pollution par quartile
    try:
        if 'Revenu_moyen_par_foyer' in df_complet.columns and 'Indice_pollution_composite' in df_complet.columns:
            # Créer les quartiles si nécessaire
            if 'Quartile_revenu' not in df_complet.columns:
                df_complet['Quartile_revenu'] = pd.qcut(df_complet['Revenu_moyen_par_foyer'],
                                                        q=4,
                                                        labels=['Q1', 'Q2', 'Q3', 'Q4'])

            # Calculer la moyenne de pollution par quartile
            pollution_par_quartile = df_complet.groupby('Quartile_revenu')['Indice_pollution_composite'].mean()
            print("Pollution moyenne par quartile de revenu calculée")

            # Afficher les résultats
            for quartile, valeur in pollution_par_quartile.items():
                print(f"Quartile {quartile}: {valeur:.4f}")
        else:
            print("AVERTISSEMENT: Colonnes nécessaires manquantes pour l'analyse par quartile")
            pollution_par_quartile = None
    except Exception as e:
        print(f"ERREUR lors de l'analyse par quartile: {e}")
        pollution_par_quartile = None

    return stats_desc, correlations, pollution_par_quartile


# --- 3. ANALYSE STATISTIQUE ---

def analyse_statistique(df_complet):
    """
    Effectue une analyse statistique plus poussée
    """
    print("Analyse statistique...")

    # Vérifier que l'indice de pollution composite est présent
    if 'Indice_pollution_composite' not in df_complet.columns:
        print("ERREUR: La colonne 'Indice_pollution_composite' est absente")
        return {
            'regression': {
                'pente': None,
                'ordonnee': None,
                'r_carre': None,
                'p_value': None,
                'erreur_std': None
            },
            'anova': {
                'f_stat': None,
                'p_value': None
            }
        }

    # Vérifier que nous avons suffisamment de données
    if df_complet['Indice_pollution_composite'].count() < 5 or df_complet['Revenu_moyen_par_foyer'].count() < 5:
        print("ERREUR: Pas assez de données pour l'analyse statistique")
        return {
            'regression': {
                'pente': None,
                'ordonnee': None,
                'r_carre': None,
                'p_value': None,
                'erreur_std': None
            },
            'anova': {
                'f_stat': None,
                'p_value': None
            }
        }

    try:
        # Régression linéaire entre revenu et pollution
        X = df_complet['Revenu_moyen_par_foyer'].values
        y = df_complet['Indice_pollution_composite'].values

        # Filtrer les valeurs NaN
        mask = ~np.isnan(X) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        # S'assurer qu'il reste suffisamment de données
        if len(X_clean) < 5:
            print("ERREUR: Pas assez de données valides après filtrage des NaN")
            return {
                'regression': {
                    'pente': None,
                    'ordonnee': None,
                    'r_carre': None,
                    'p_value': None,
                    'erreur_std': None
                },
                'anova': {
                    'f_stat': None,
                    'p_value': None
                }
            }

        # Régression
        X_reshape = X_clean.reshape(-1, 1)  # Reshape pour sklearn
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_clean, y_clean)

        print(f"Régression: y = {slope:.6f}x + {intercept:.6f}, r² = {r_value ** 2:.4f}, p = {p_value:.4f}")

        # Test ANOVA entre quartiles de revenu et pollution
        from scipy.stats import f_oneway

        # S'assurer que la colonne de quartiles existe
        if 'Quartile_revenu' not in df_complet.columns:
            print("Création des quartiles de revenu...")
            df_complet['Quartile_revenu'] = pd.qcut(df_complet['Revenu_moyen_par_foyer'],
                                                    q=4,
                                                    labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # Grouper les données par quartile pour l'ANOVA
        groupes = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            groupe = df_complet[df_complet['Quartile_revenu'] == q]['Indice_pollution_composite'].dropna()
            if len(groupe) > 0:
                groupes.append(groupe)

        # S'assurer qu'il y a des données dans au moins deux groupes
        if len(groupes) > 1:
            f_stat, anova_p_value = f_oneway(*groupes)
            print(f"ANOVA: F = {f_stat:.4f}, p = {anova_p_value:.4f}")
        else:
            print("ATTENTION: Pas assez de groupes pour l'ANOVA")
            f_stat, anova_p_value = None, None

        resultats = {
            'regression': {
                'pente': slope,
                'ordonnee': intercept,
                'r_carre': r_value ** 2,
                'p_value': p_value,
                'erreur_std': std_err
            },
            'anova': {
                'f_stat': f_stat,
                'p_value': anova_p_value
            }
        }

        return resultats

    except Exception as e:
        print(f"ERREUR dans l'analyse statistique: {e}")
        import traceback
        traceback.print_exc()

        return {
            'regression': {
                'pente': None,
                'ordonnee': None,
                'r_carre': None,
                'p_value': None,
                'erreur_std': None
            },
            'anova': {
                'f_stat': None,
                'p_value': None
            }
        }


# --- 4. PRÉPARATION POUR VISUALISATION ET EXPORT ---

def preparer_pour_visualisation(df_complet):
    """
    Prépare les données pour la visualisation et l'export
    """
    print("Préparation des données pour visualisation...")

    if df_complet.empty:
        print("ERREUR: DataFrame vide, impossible de préparer pour visualisation")
        return pd.DataFrame()

    # Vérifier les colonnes disponibles
    colonnes_disponibles = df_complet.columns.tolist()
    print(f"Colonnes disponibles: {colonnes_disponibles}")

    # Définir les colonnes à inclure dans la visualisation
    colonnes_a_inclure = ['INSEE_COM']

    # Ajouter le nom de la commune si disponible
    if 'Libellé de la commune' in colonnes_disponibles:
        colonnes_a_inclure.append('Libellé de la commune')

    # Ajouter les colonnes de pollution
    colonnes_pollution = [col for col in colonnes_disponibles if col.startswith('Moyenne_')]
    colonnes_a_inclure.extend(colonnes_pollution)

    # Ajouter l'indice composite
    if 'Indice_pollution_composite' in colonnes_disponibles:
        colonnes_a_inclure.append('Indice_pollution_composite')

    # Ajouter les indicateurs économiques
    for col in ['Revenu_moyen_par_foyer', 'Pourcentage_foyers_imposes']:
        if col in colonnes_disponibles:
            colonnes_a_inclure.append(col)

    # Ajouter les quartiles
    if 'Quartile_revenu' in colonnes_disponibles:
        colonnes_a_inclure.append('Quartile_revenu')

    # Sélectionner les colonnes disponibles
    colonnes_finales = [col for col in colonnes_a_inclure if col in colonnes_disponibles]
    df_visualisation = df_complet[colonnes_finales].copy()

    # Créer un indice de justice environnementale si possible
    if 'Indice_pollution_composite' in df_visualisation.columns and 'Revenu_moyen_par_foyer' in df_visualisation.columns:
        # Éviter la division par zéro
        denominateur = df_visualisation['Revenu_moyen_par_foyer'].replace(0, np.nan) / 10000
        df_visualisation['Indice_justice_env'] = df_visualisation['Indice_pollution_composite'] / denominateur
        print("Indice de justice environnementale calculé")

    print(
        f"Données préparées pour visualisation: {df_visualisation.shape[0]} communes, {df_visualisation.shape[1]} colonnes")

    return df_visualisation


def exporter_donnees(df_visualisation, annee, format_sortie='csv'):
    """
    Exporte les données traitées dans différents formats

    Parameters:
    -----------
    df_visualisation : DataFrame
        DataFrame contenant les données préparées pour la visualisation
    annee : int
        Année de l'analyse
    format_sortie : str, optional
        Format de sortie ('csv', 'geojson', 'tous')
    """
    print("Exportation des données...")

    if df_visualisation.empty:
        print("ERREUR: Aucune donnée à exporter")
        return

    # Créer un dossier pour les résultats s'il n'existe pas
    if not os.path.exists('resultats'):
        os.makedirs('resultats')

    # Exporter en CSV
    if format_sortie in ['csv', 'tous']:
        fichier_csv = f'resultats/analyse_pollution_revenus_{annee}.csv'
        df_visualisation.to_csv(fichier_csv, index=False)
        print(f"Données exportées au format CSV: {fichier_csv}")

    # Exporter en GeoJSON (nécessiterait des données géospatiales)
    if format_sortie in ['geojson', 'tous']:
        print("Note: L'export en GeoJSON nécessiterait de joindre avec des données de frontières communales")
        print("      Cette fonctionnalité sera implémentée ultérieurement")

    print(f"Exportation terminée pour l'année {annee}")


# Fonction pour gérer plusieurs années d'analyse
def analyser_multi_annees(dossier_ircom, dossier_pollution, annees):
    """
    Analyse les données pour plusieurs années et compare les résultats

    Parameters:
    -----------
    dossier_ircom : str
        Chemin du dossier contenant les données IRCOM
    dossier_pollution : str
        Chemin du dossier contenant les sous-dossiers de pollution (No2, NOx, etc.)
    annees : list
        Liste des années à analyser

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats pour chaque année
    """
    print(f"Analyse multi-années pour les années: {annees}")

    resultats_par_annee = {}

    for annee in annees:
        print(f"\n{'=' * 50}\nAnalyse pour l'année {annee}\n{'=' * 50}")

        resultats = analyser_pollution_revenus(dossier_ircom, dossier_pollution, annee)

        if resultats is not None:
            resultats_par_annee[annee] = resultats
            print(f"Analyse pour {annee} terminée avec succès.")
        else:
            print(f"ÉCHEC de l'analyse pour {annee}")

    # Si plusieurs années ont été analysées avec succès, créer une analyse comparative
    if len(resultats_par_annee) > 1:
        print("\nAnalyse comparative des années...")

        # Créer un dossier pour les résultats s'il n'existe pas
        if not os.path.exists('resultats'):
            os.makedirs('resultats')

        # Comparer les corrélations au fil du temps
        correlations_temporelles = {}

        for annee, res in resultats_par_annee.items():
            if 'correlations' in res and res['correlations'] is not None:
                for col_pol in [c for c in res['correlations'].columns if c.startswith('Moyenne_')]:
                    if 'Revenu_moyen_par_foyer' in res['correlations'].index:
                        if col_pol not in correlations_temporelles:
                            correlations_temporelles[col_pol] = {}

                        corr = res['correlations'].loc['Revenu_moyen_par_foyer', col_pol]
                        correlations_temporelles[col_pol][annee] = corr

        # Exporter les corrélations temporelles
        if correlations_temporelles:
            df_corr_temp = pd.DataFrame(correlations_temporelles)
            df_corr_temp.to_csv('resultats/correlations_temporelles.csv')
            print("Corrélations temporelles exportées dans 'resultats/correlations_temporelles.csv'")

        # Comparer les indices de pollution par quartile de revenu au fil du temps
        pollution_par_quartile_temporelle = {}

        for annee, res in resultats_par_annee.items():
            if 'pollution_par_quartile' in res and res['pollution_par_quartile'] is not None:
                for quartile in res['pollution_par_quartile'].index:
                    if quartile not in pollution_par_quartile_temporelle:
                        pollution_par_quartile_temporelle[quartile] = {}

                    pollution_par_quartile_temporelle[quartile][annee] = res['pollution_par_quartile'][quartile]

        # Exporter la pollution par quartile temporelle
        if pollution_par_quartile_temporelle:
            df_poll_quart_temp = pd.DataFrame(pollution_par_quartile_temporelle)
            df_poll_quart_temp.to_csv('resultats/pollution_par_quartile_temporelle.csv')
            print("Pollution par quartile temporelle exportée dans 'resultats/pollution_par_quartile_temporelle.csv'")

    return resultats_par_annee


# --- 5. FONCTION PRINCIPALE ---

def analyser_pollution_revenus(dossier_ircom, dossier_pollution, annee):
    """
    Fonction principale qui orchestre l'analyse complète

    Parameters:
    -----------
    dossier_ircom : str
        Chemin du dossier contenant les données IRCOM
    dossier_pollution : str
        Chemin du dossier contenant les sous-dossiers de pollution (No2, NOx, etc.)
    annee : int
        Année à analyser
    """
    print(f"Analyse des données pour l'année {annee}...")

    # 1. Chargement des données de pollution
    print("Étape 1: Chargement des données de pollution")
    dataframes_pollution = {}

    # Tentative de chargement pour chaque type de polluant
    types_polluants = {
        'NO2': f"{dossier_pollution}/No2",
        'NOx': f"{dossier_pollution}/NOx",
        'PM10': f"{dossier_pollution}/PM10",
        'PM25': f"{dossier_pollution}/PM25"
    }

    for polluant, chemin in types_polluants.items():
        try:
            df = charger_donnees_pollution(chemin, annee)
            if not df.empty:
                dataframes_pollution[polluant] = df
                print(f"Données pour {polluant} chargées avec succès!")
            else:
                print(f"Aucune donnée trouvée pour {polluant} en {annee}")
        except Exception as e:
            print(f"Erreur lors du chargement des données {polluant}: {e}")

    if not dataframes_pollution:
        print(f"ERREUR: Aucune donnée de pollution n'a pu être chargée pour {annee}")
        return None

    # Charger les données fiscales
    print("Étape 1 bis: Chargement des données fiscales")
    df_fiscales = charger_donnees_fiscales(f"{dossier_ircom}/IRCOM_{annee}.csv")

    if df_fiscales is None or df_fiscales.empty:
        print("ERREUR: Données fiscales non disponibles")
        return None

    # 2. Prétraitement des données
    print("Étape 2: Prétraitement des données")
    dataframes_pretraites = {}
    for polluant, df in dataframes_pollution.items():
        df_pretraite, _, _, _ = pretraiter_donnees_pollution(df, polluant)
        if not df_pretraite.empty:
            dataframes_pretraites[polluant] = df_pretraite
            print(f"Prétraitement réussi pour {polluant}")
        else:
            print(f"Prétraitement échoué pour {polluant}")

    if not dataframes_pretraites:
        print("ERREUR: Aucune donnée de pollution n'a pu être prétraitée correctement")
        return None

    # 3. Fusion des données
    print("Étape 3: Fusion des données")

    # Initialiser avec le premier DataFrame prétraité
    premier_polluant = list(dataframes_pretraites.keys())[0]
    df_complet = dataframes_pretraites[premier_polluant].copy()

    # Fusionner avec les autres DataFrames de pollution
    for polluant, df in dataframes_pretraites.items():
        if polluant != premier_polluant:  # Éviter de fusionner avec soi-même
            df_complet = df_complet.merge(df, on='INSEE_COM', how='outer')

    # Fusionner avec les données fiscales
    df_complet = df_complet.merge(df_fiscales, on='INSEE_COM', how='inner')

    if df_complet.empty:
        print("ERREUR: Aucune correspondance trouvée entre données de pollution et données fiscales")
        return None

    # Créer un indice composite de pollution avec les polluants disponibles
    colonnes_pollution = [col for col in df_complet.columns if col.startswith('Moyenne_')]
    if colonnes_pollution:
        print(
            f"Calcul de l'indice de pollution composite basé sur {len(colonnes_pollution)} polluants: {colonnes_pollution}")
        df_complet['Indice_pollution_composite'] = df_complet[colonnes_pollution].mean(axis=1, skipna=True)
    else:
        print("ERREUR: Aucune colonne de pollution trouvée pour calculer l'indice composite")
        return None

    # 4. Analyse exploratoire
    print("Étape 4: Analyse exploratoire")
    stats_desc, correlations, pollution_par_quartile = analyse_exploratoire(df_complet)

    # 5. Analyse statistique
    print("Étape 5: Analyse statistique")
    resultats_stats = analyse_statistique(df_complet)

    # 6. Préparation pour visualisation
    print("Étape 6: Préparation pour visualisation")
    df_visualisation = preparer_pour_visualisation(df_complet)

    # 7. Export
    print("Étape 7: Export des données")
    exporter_donnees(df_visualisation, annee)

    print(f"Analyse pour {annee} terminée avec succès!")

    return {
        'donnees_completes': df_complet,
        'donnees_visualisation': df_visualisation,
        'statistiques': stats_desc,
        'correlations': correlations,
        'pollution_par_quartile': pollution_par_quartile,
        'resultats_statistiques': resultats_stats
    }


# --- EXEMPLE D'UTILISATION ---

def main():
    """
    Fonction principale d'exécution
    """
    # Définir les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Analyse du lien entre richesse et pollution atmosphérique')
    parser.add_argument('--airparif', type=str, default='./data/air-quality/idf/final',
                        help='Dossier contenant les données AirParif')
    parser.add_argument('--ircom', type=str, default='./data/revenus-fiscaux/IRCOM/clean/',
                        help='Dossier contenant les données IRCOM')
    parser.add_argument('--output', type=str, default='resultats',
                        help='Dossier de sortie pour les résultats')
    parser.add_argument('--years', type=int, nargs='+', default=[2018, 2019, 2020, 2021, 2022, 2023],
                        help='Années à analyser')
    parser.add_argument('--year', type=int, help='Analyser une année spécifique')
    parser.add_argument('--all', action='store_true', help='Analyser toutes les années disponibles')

    args = parser.parse_args()

    # Définition des chemins de données
    DOSSIER_AIRPARIF = args.airparif
    DOSSIER_IRCOM = args.ircom
    DOSSIER_OUTPUT = args.output

    print("Analyse du lien entre richesse et pollution atmosphérique")
    print("=========================================================")
    print(f"Dossier AirParif: {DOSSIER_AIRPARIF}")
    print(f"Dossier IRCOM: {DOSSIER_IRCOM}")
    print(f"Dossier de sortie: {DOSSIER_OUTPUT}")

    # Vérifier l'existence des dossiers
    if not os.path.exists(DOSSIER_AIRPARIF):
        print(f"ERREUR: Le dossier {DOSSIER_AIRPARIF} n'existe pas.")
        sys.exit(1)

    if not os.path.exists(DOSSIER_IRCOM):
        print(f"ERREUR: Le dossier {DOSSIER_IRCOM} n'existe pas.")
        sys.exit(1)

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(DOSSIER_OUTPUT, exist_ok=True)

    # Vérifier la structure des dossiers de pollution
    for polluant in ["No2", "NOx", "PM10", "PM25"]:
        dossier_polluant = os.path.join(DOSSIER_AIRPARIF, polluant)
        if not os.path.exists(dossier_polluant):
            print(f"AVERTISSEMENT: Le dossier {dossier_polluant} n'existe pas.")

    # Déterminer les années à analyser
    if args.year:
        # Analyser une seule année spécifique
        annees = [args.year]
    elif args.all:
        # Analyser toutes les années pour lesquelles on trouve des données
        annees = []
        # Chercher dans IRCOM
        for item in os.listdir(DOSSIER_IRCOM):
            if os.path.isdir(os.path.join(DOSSIER_IRCOM, item)):
                try:
                    annee = int(item)
                    annees.append(annee)
                except ValueError:
                    pass
            else:
                # Essayer de détecter l'année dans le nom du fichier
                for an in range(2010, 2025):
                    if str(an) in item:
                        annees.append(an)
                        break

        # Chercher aussi dans AirParif
        for polluant in ["No2", "NOx", "PM10", "PM25"]:
            dossier_polluant = os.path.join(DOSSIER_AIRPARIF, polluant)
            if os.path.exists(dossier_polluant):
                for fichier in os.listdir(dossier_polluant):
                    for an in range(2010, 2025):
                        if str(an) in fichier:
                            annees.append(an)
                            break

        # Éliminer les doublons et trier
        annees = sorted(list(set(annees)))
    else:
        # Utiliser les années spécifiées dans --years
        annees = args.years

    print(f"Années à analyser: {annees}")

    if len(annees) == 0:
        print("ERREUR: Aucune année à analyser")
        sys.exit(1)

    if len(annees) == 1:
        # Analyser une seule année
        print(f"\nAnalyse pour l'année {annees[0]}")
        print("-" * 30)
        resultats = analyser_pollution_revenus(DOSSIER_IRCOM, DOSSIER_AIRPARIF, annees[0])

        if resultats:
            print(f"Analyse pour {annees[0]} terminée avec succès.")

            # Afficher quelques résultats
            if 'correlations' in resultats and resultats['correlations'] is not None:
                print("\nRésultats clés - Corrélations:")
                colonnes_pollution = [col for col in resultats['correlations'].columns if col.startswith('Moyenne_')]
                for col_pol in colonnes_pollution:
                    if 'Revenu_moyen_par_foyer' in resultats['correlations'].index:
                        corr = resultats['correlations'].loc['Revenu_moyen_par_foyer', col_pol]
                        print(f"  * {col_pol} vs Revenu moyen: {corr:.3f}")

            if 'pollution_par_quartile' in resultats and resultats['pollution_par_quartile'] is not None:
                print("\nRésultats clés - Pollution par quartile:")
                for quartile, valeur in resultats['pollution_par_quartile'].items():
                    print(f"  * {quartile}: {valeur:.2f}")

            if 'resultats_statistiques' in resultats and 'regression' in resultats['resultats_statistiques']:
                reg = resultats['resultats_statistiques']['regression']
                if all(v is not None for v in reg.values()):
                    print("\nRésultats clés - Régression linéaire:")
                    print(f"  * R² = {reg['r_carre']:.3f}, p-value = {reg['p_value']:.4f}")
                    print(f"  * Équation: Pollution = {reg['pente']:.4f} × Revenu + {reg['ordonnee']:.4f}")
        else:
            print(f"ERREUR: L'analyse pour {annees[0]} a échoué.")
    else:
        # Analyser plusieurs années
        resultats_par_annee = analyser_multi_annees(DOSSIER_IRCOM, DOSSIER_AIRPARIF, annees)

        if resultats_par_annee:
            print(f"\nAnalyses terminées pour {len(resultats_par_annee)} années sur {len(annees)} demandées.")

            # Créer les visualisations des tendances
            print("\nCréation des visualisations des tendances temporelles...")

            # Si au moins deux années ont été analysées avec succès
            if len(resultats_par_annee) >= 2:
                # Visualisation de l'évolution des corrélations
                correlations_temporelles = {}

                for annee, res in resultats_par_annee.items():
                    if 'correlations' in res and res['correlations'] is not None:
                        for col_pol in [c for c in res['correlations'].columns if c.startswith('Moyenne_')]:
                            if 'Revenu_moyen_par_foyer' in res['correlations'].index:
                                if col_pol not in correlations_temporelles:
                                    correlations_temporelles[col_pol] = {}

                                corr = res['correlations'].loc['Revenu_moyen_par_foyer', col_pol]
                                correlations_temporelles[col_pol][annee] = corr

                if correlations_temporelles:
                    # Tracer l'évolution des corrélations
                    plt.figure(figsize=(10, 6))
                    for polluant, corrs in correlations_temporelles.items():
                        plt.plot(list(corrs.keys()), list(corrs.values()), marker='o', label=polluant)

                    plt.title('Évolution des corrélations entre pollution et revenu')
                    plt.xlabel('Année')
                    plt.ylabel('Coefficient de corrélation')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.savefig(os.path.join(DOSSIER_OUTPUT, 'evolution_correlations.png'))
                    print(
                        f"Graphique d'évolution des corrélations enregistré dans {DOSSIER_OUTPUT}/evolution_correlations.png")

                # Visualisation de l'évolution de la pollution par quartile
                pollution_par_quartile_temporelle = {}

                for annee, res in resultats_par_annee.items():
                    if 'pollution_par_quartile' in res and res['pollution_par_quartile'] is not None:
                        for quartile in res['pollution_par_quartile'].index:
                            if quartile not in pollution_par_quartile_temporelle:
                                pollution_par_quartile_temporelle[quartile] = {}

                            pollution_par_quartile_temporelle[quartile][annee] = res['pollution_par_quartile'][quartile]

                if pollution_par_quartile_temporelle:
                    # Tracer l'évolution de la pollution par quartile
                    plt.figure(figsize=(10, 6))
                    for quartile, poll in pollution_par_quartile_temporelle.items():
                        plt.plot(list(poll.keys()), list(poll.values()), marker='o', label=f'Quartile {quartile}')

                    plt.title('Évolution de la pollution par quartile de revenu')
                    plt.xlabel('Année')
                    plt.ylabel('Indice de pollution composite')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.savefig(os.path.join(DOSSIER_OUTPUT, 'evolution_pollution_quartile.png'))
                    print(
                        f"Graphique d'évolution de la pollution par quartile enregistré dans {DOSSIER_OUTPUT}/evolution_pollution_quartile.png")
        else:
            print("ERREUR: Aucune analyse n'a réussi.")

    print("\nTraitement terminé!")


if __name__ == "__main__":
    main()