"""
Visualisation du lien entre richesse et pollution atmosphérique
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
import traceback


def visualiser_correlation(df, fichier_sortie=None):
    """
    Créer un nuage de points avec régression pour visualiser la corrélation
    entre richesse et pollution
    """
    try:
        # Filtrer les données pour n'avoir que les lignes complètes pour les deux colonnes
        df_filtered = df.dropna(subset=['Revenu_moyen_par_foyer', 'Indice_pollution_composite'])

        if len(df_filtered) < 3:
            print("ERREUR: Pas assez de données complètes pour calculer la corrélation")
            return

        print(f"Données filtrées pour la corrélation: {len(df_filtered)} communes")

        plt.figure(figsize=(12, 8))

        # Créer un nuage de points
        sns.regplot(
            x='Revenu_moyen_par_foyer',
            y='Indice_pollution_composite',
            data=df_filtered,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )

        # Calculer le coefficient de corrélation
        try:
            corr, p_value = stats.pearsonr(
                df_filtered['Revenu_moyen_par_foyer'].values,
                df_filtered['Indice_pollution_composite'].values
            )

            # Ajouter le titre et les labels
            plt.title(
                f'Relation entre niveau de revenu et pollution\nCoefficient de corrélation: {corr:.3f} (p-value: {p_value:.3f})',
                fontsize=14)
        except Exception as e:
            print(f"Erreur lors du calcul de la corrélation: {e}")
            plt.title('Relation entre niveau de revenu et pollution', fontsize=14)

        plt.xlabel('Revenu moyen par foyer (€)', fontsize=12)
        plt.ylabel('Indice de pollution composite', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Ajouter des annotations pour certaines communes
        try:
            if 'Libellé de la commune' in df_filtered.columns:
                # Identifier quelques communes intéressantes
                df_sorted = df_filtered.sort_values('Indice_pollution_composite', ascending=False)
                communes_elevees = df_sorted.head(3)  # Communes avec plus forte pollution

                df_sorted = df_filtered.sort_values('Revenu_moyen_par_foyer', ascending=False)
                communes_riches = df_sorted.head(3)  # Communes les plus riches

                # Annoter les communes
                communes_a_annoter = pd.concat([communes_elevees, communes_riches]).drop_duplicates()

                for idx, commune in communes_a_annoter.iterrows():
                    plt.annotate(
                        commune['Libellé de la commune'],
                        xy=(commune['Revenu_moyen_par_foyer'], commune['Indice_pollution_composite']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
        except Exception as e:
            print(f"Erreur lors de l'annotation des communes: {e}")

        plt.tight_layout()

        if fichier_sortie:
            plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {fichier_sortie}")
        else:
            plt.show()

        plt.close()

        return True
    except Exception as e:
        print(f"Erreur lors de la création du nuage de points: {e}")
        traceback.print_exc()
        return False


def visualiser_quartiles_revenu(df, fichier_sortie=None):
    """
    Créer un graphique à barres montrant les niveaux de pollution par quartile de revenu
    """
    try:
        # Filtrer les données pour n'avoir que les lignes complètes
        df_filtered = df.dropna(subset=['Revenu_moyen_par_foyer', 'Indice_pollution_composite'])

        if len(df_filtered) < 4:  # Besoin d'au moins quelques points par quartile
            print("ERREUR: Pas assez de données complètes pour l'analyse par quartile")
            return False

        print(f"Données filtrées pour l'analyse par quartile: {len(df_filtered)} communes")

        plt.figure(figsize=(10, 6))

        # S'assurer que la colonne quartile existe ou la créer
        try:
            if 'Quartile_revenu' not in df_filtered.columns:
                print("Création des quartiles de revenu...")

                # Déterminer le nombre de quartiles en fonction des données disponibles
                n_quartiles = min(4, max(2, len(df_filtered) // 3))

                # Créer les labels appropriés
                if n_quartiles == 2:
                    labels = ['Q1 (plus pauvre)', 'Q2 (plus riche)']
                elif n_quartiles == 3:
                    labels = ['Q1 (plus pauvre)', 'Q2 (moyen)', 'Q3 (plus riche)']
                else:
                    labels = ['Q1 (plus pauvre)', 'Q2', 'Q3', 'Q4 (plus riche)']

                try:
                    # Tenter avec qcut (pour des quartiles de taille égale)
                    df_filtered['Quartile_revenu'] = pd.qcut(
                        df_filtered['Revenu_moyen_par_foyer'],
                        q=n_quartiles,
                        labels=labels
                    )
                except ValueError:
                    # Si qcut échoue (valeurs dupliquées), utiliser cut (intervalles égaux)
                    df_filtered['Quartile_revenu'] = pd.cut(
                        df_filtered['Revenu_moyen_par_foyer'],
                        bins=n_quartiles,
                        labels=labels
                    )
        except Exception as e:
            print(f"Erreur lors de la création des quartiles: {e}")
            return False

        # Calculer la moyenne de pollution par quartile
        try:
            pollution_par_quartile = df_filtered.groupby('Quartile_revenu')[
                'Indice_pollution_composite'].mean().reset_index()

            # Vérifier qu'il y a des données dans chaque quartile
            if pollution_par_quartile.empty:
                print("ERREUR: Impossible de calculer la pollution par quartile")
                return False

            print(f"Nombre de quartiles: {len(pollution_par_quartile)}")

            # Calculer les intervalles de confiance si possible
            has_errors = False
            if len(df_filtered) >= 8:  # Au moins quelques points par groupe pour un intervalle significatif
                grouped = df_filtered.groupby('Quartile_revenu')['Indice_pollution_composite']
                try:
                    errors = grouped.sem().mul(1.96)  # Intervalle de confiance à 95%
                    has_errors = True
                except:
                    has_errors = False
                    print("Impossible de calculer les intervalles de confiance")
        except Exception as e:
            print(f"Erreur lors du calcul de la pollution par quartile: {e}")
            return False

        # Créer le graphique à barres
        ax = sns.barplot(x='Quartile_revenu', y='Indice_pollution_composite', data=pollution_par_quartile,
                         palette='viridis', alpha=0.8)

        # Ajouter les barres d'erreur si disponibles
        if has_errors:
            plt.errorbar(
                x=np.arange(len(errors)),
                y=pollution_par_quartile['Indice_pollution_composite'],
                yerr=errors.values,
                fmt='none',
                capsize=5,
                color='black'
            )

        # Ajouter les valeurs sur les barres
        for i, v in enumerate(pollution_par_quartile['Indice_pollution_composite']):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold')

        plt.title('Indice de pollution moyenne par quartile de revenu', fontsize=14)
        plt.xlabel('Quartile de revenu', fontsize=12)
        plt.ylabel('Indice de pollution composite', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')

        plt.tight_layout()

        if fichier_sortie:
            plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {fichier_sortie}")
        else:
            plt.show()

        plt.close()
        return True

    except Exception as e:
        print(f"Erreur lors de la création du graphique par quartile: {e}")
        traceback.print_exc()
        return False


def visualiser_heatmap_correlations(df, fichier_sortie=None):
    """
    Créer une heatmap des corrélations entre différentes variables
    """
    try:
        # Sélectionner les colonnes pertinentes
        colonnes_pollution = [col for col in df.columns if col.startswith('Moyenne_')]
        colonnes_economiques = []

        # Vérifier quelles colonnes économiques sont disponibles
        for col in ['Revenu_moyen_par_foyer', 'Pourcentage_foyers_imposes']:
            if col in df.columns:
                colonnes_economiques.append(col)

        # Filtrer pour n'inclure que les colonnes disponibles
        if 'Indice_pollution_composite' in df.columns:
            colonnes_pollution.append('Indice_pollution_composite')

        colonnes_analyse = colonnes_pollution + colonnes_economiques

        if len(colonnes_analyse) < 2:
            print("ERREUR: Pas assez de colonnes pour calculer les corrélations")
            return False

        # Filtrer les données pour n'avoir que les lignes avec des données pour toutes les colonnes
        df_filtered = df[colonnes_analyse].copy()

        # Calculer le nombre de valeurs non-NA par colonne
        na_counts = df_filtered.isna().sum()
        print("Valeurs manquantes par colonne:")
        for col in colonnes_analyse:
            print(f"- {col}: {na_counts[col]} valeurs manquantes sur {len(df_filtered)}")

        # Si toutes les colonnes ont trop de valeurs manquantes, essayer de ne garder que les plus complètes
        if (na_counts > 0.7 * len(df_filtered)).any():
            print("AVERTISSEMENT: Certaines colonnes ont beaucoup de valeurs manquantes")
            # Garder uniquement les colonnes les plus complètes
            colonnes_utilisables = [col for col in colonnes_analyse if na_counts[col] < 0.5 * len(df_filtered)]

            if len(colonnes_utilisables) < 2:
                print("ERREUR: Pas assez de colonnes utilisables pour la heatmap")
                return False

            print(f"Utilisation de {len(colonnes_utilisables)} colonnes sur {len(colonnes_analyse)} disponibles")
            df_filtered = df[colonnes_utilisables].copy()
            colonnes_analyse = colonnes_utilisables

        # Calculer le nombre de valeurs complètes (toutes les colonnes ont une valeur)
        df_complete = df_filtered.dropna()
        if len(df_complete) < 3:
            print(f"AVERTISSEMENT: Seulement {len(df_complete)} lignes complètes sur {len(df_filtered)}")

            # Stratégie alternative: calculer la corrélation par paires
            print("Calcul de la corrélation par paires à la place")
            corr_matrix = df_filtered.corr(method='pearson', min_periods=3)
        else:
            # Utiliser les lignes complètes
            df_filtered = df_complete
            print(f"Données filtrées pour la heatmap: {len(df_filtered)} communes, {len(colonnes_analyse)} variables")
            corr_matrix = df_filtered.corr()

        plt.figure(figsize=(10, 8))

        # Masquer la diagonale supérieure pour plus de lisibilité
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Créer la heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            fmt=".2f"
        )

        plt.title('Matrice de corrélation entre pollution et indicateurs socio-économiques', fontsize=14)
        plt.tight_layout()

        if fichier_sortie:
            plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {fichier_sortie}")
        else:
            plt.show()

        plt.close()
        return True

    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {e}")
        traceback.print_exc()
        return False


def create_all_visualizations(fichier_donnees, dossier_sortie='visualisations', fichier_geojson=None):
    """
    Crée toutes les visualisations à partir d'un fichier de données
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)

    # Charger les données
    try:
        df = pd.read_csv(fichier_donnees)
        print(f"Données chargées: {df.shape[0]} communes, {df.shape[1]} colonnes")

        # Afficher quelques informations sur les données
        print("\nAperçu des colonnes disponibles:")
        for col in df.columns:
            non_na = df[col].count()
            print(f"- {col}: {non_na}/{len(df)} valeurs non-NA ({non_na / len(df) * 100:.1f}%)")

        # Vérifier les valeurs manquantes pour les colonnes principales
        colonnes_cles = ['Revenu_moyen_par_foyer', 'Indice_pollution_composite']
        colonnes_presentes = [col for col in colonnes_cles if col in df.columns]

        if len(colonnes_presentes) < 2:
            print(f"ERREUR: Colonnes requises manquantes. Colonnes disponibles: {df.columns.tolist()}")
            print(f"Colonnes requises: {colonnes_cles}")
            return

        df_complete = df.dropna(subset=colonnes_presentes)
        print(
            f"\nNombre de communes avec données complètes: {len(df_complete)}/{len(df)} ({len(df_complete) / len(df) * 100:.1f}%)")

    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        traceback.print_exc()
        return

    # Vérifier que nous avons les colonnes nécessaires
    colonnes_requises = ['Revenu_moyen_par_foyer', 'Indice_pollution_composite']
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]

    if colonnes_manquantes:
        print(f"ERREUR: Colonnes requises manquantes: {colonnes_manquantes}")
        print(f"Colonnes disponibles: {df.columns.tolist()}")
        return

    # 1. Visualisation de la corrélation
    print("\n1. Création du nuage de points...")
    success1 = visualiser_correlation(df, os.path.join(dossier_sortie, 'correlation_revenu_pollution.png'))

    # 2. Visualisation par quartile
    print("\n2. Création du graphique par quartile...")
    success2 = visualiser_quartiles_revenu(df, os.path.join(dossier_sortie, 'pollution_par_quartile.png'))

    # 3. Heatmap des corrélations
    print("\n3. Création de la heatmap des corrélations...")
    success3 = visualiser_heatmap_correlations(df, os.path.join(dossier_sortie, 'heatmap_correlations.png'))

    # Création d'un rapport textuel si au moins une visualisation a réussi
    if success1 or success2 or success3:
        try:
            with open(os.path.join(dossier_sortie, 'rapport_analyse.txt'), 'w') as f:
                f.write("RAPPORT D'ANALYSE - LIEN ENTRE RICHESSE ET POLLUTION\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Données analysées: {fichier_donnees}\n")
                f.write(f"Nombre de communes: {len(df)}\n\n")

                # Statistiques sur les revenus
                f.write("STATISTIQUES DE REVENU\n")
                f.write("-" * 30 + "\n")
                revenu_stats = df['Revenu_moyen_par_foyer'].describe()
                f.write(f"Revenu moyen minimum: {revenu_stats['min']:.2f} €\n")
                f.write(f"Revenu moyen maximum: {revenu_stats['max']:.2f} €\n")
                f.write(f"Revenu moyen médian: {revenu_stats['50%']:.2f} €\n\n")

                # Statistiques sur la pollution
                f.write("STATISTIQUES DE POLLUTION\n")
                f.write("-" * 30 + "\n")
                poll_stats = df['Indice_pollution_composite'].describe()
                f.write(f"Indice de pollution minimum: {poll_stats['min']:.2f}\n")
                f.write(f"Indice de pollution maximum: {poll_stats['max']:.2f}\n")
                f.write(f"Indice de pollution médian: {poll_stats['50%']:.2f}\n\n")

                # Coefficients de corrélation
                if success1:
                    f.write("CORRÉLATIONS\n")
                    f.write("-" * 30 + "\n")
                    # Filtrer pour éviter les NA
                    df_filtered = df.dropna(subset=['Revenu_moyen_par_foyer', 'Indice_pollution_composite'])
                    if len(df_filtered) >= 3:
                        try:
                            corr, p_value = stats.pearsonr(
                                df_filtered['Revenu_moyen_par_foyer'].values,
                                df_filtered['Indice_pollution_composite'].values
                            )
                            f.write(f"Corrélation revenu-pollution: {corr:.3f} (p-value: {p_value:.3f})\n")

                            # Interprétation
                            f.write("\nINTERPRÉTATION\n")
                            f.write("-" * 30 + "\n")

                            if abs(corr) < 0.2:
                                interpretation = "Très faible corrélation ou absence de corrélation"
                            elif abs(corr) < 0.4:
                                interpretation = "Faible corrélation"
                            elif abs(corr) < 0.6:
                                interpretation = "Corrélation modérée"
                            elif abs(corr) < 0.8:
                                interpretation = "Forte corrélation"
                            else:
                                interpretation = "Très forte corrélation"

                            if corr < 0:
                                sens = "négative"
                                explication = "les communes plus riches tendent à avoir moins de pollution"
                            else:
                                sens = "positive"
                                explication = "les communes plus riches tendent à avoir plus de pollution"

                            f.write(f"{interpretation} {sens} : {explication}.\n\n")

                            # Significativité
                            if p_value < 0.01:
                                f.write("Cette corrélation est statistiquement très significative (p<0.01).\n")
                            elif p_value < 0.05:
                                f.write("Cette corrélation est statistiquement significative (p<0.05).\n")
                            elif p_value < 0.1:
                                f.write("Cette corrélation est marginalement significative (p<0.1).\n")
                            else:
                                f.write("Cette corrélation n'est pas statistiquement significative (p>0.1).\n")
                        except Exception as e:
                            f.write(f"Erreur lors du calcul de la corrélation: {e}\n")
                    else:
                        f.write("Impossible de calculer les corrélations (données insuffisantes)\n")

                f.write("\nVISUALISATIONS GÉNÉRÉES\n")
                f.write("-" * 30 + "\n")
                if success1:
                    f.write(f"1. Nuage de points: {os.path.join(dossier_sortie, 'correlation_revenu_pollution.png')}\n")
                if success2:
                    f.write(
                        f"2. Pollution par quartile: {os.path.join(dossier_sortie, 'pollution_par_quartile.png')}\n")
                if success3:
                    f.write(
                        f"3. Heatmap des corrélations: {os.path.join(dossier_sortie, 'heatmap_correlations.png')}\n")

            print(f"\nRapport d'analyse créé: {os.path.join(dossier_sortie, 'rapport_analyse.txt')}")
        except Exception as e:
            print(f"Erreur lors de la création du rapport: {e}")
            traceback.print_exc()

    print(f"\nVisualisation terminée.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Créer des visualisations pour l'analyse pollution-revenus")
    parser.add_argument('--data', type=str, required=True, help='Chemin du fichier CSV avec les données analysées')
    parser.add_argument('--output', type=str, default='visualisations',
                        help='Dossier de sortie pour les visualisations')
    parser.add_argument('--geojson', type=str, help='Chemin du fichier GeoJSON pour la carte (optionnel)')
    parser.add_argument('--annee', type=str, help='Année des données (pour le nom des fichiers)')

    args = parser.parse_args()

    # Créer le dossier de sortie avec l'année si spécifiée
    dossier_sortie = args.output
    if args.annee:
        dossier_sortie = f"{args.output}_{args.annee}"

    create_all_visualizations(args.data, dossier_sortie, args.geojson)