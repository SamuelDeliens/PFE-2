"""
Visualisation avancée du lien entre richesse et pollution atmosphérique
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
import traceback


def visualiser_correlation_par_polluant(df, fichier_sortie=None):
    """
    Créer des nuages de points avec régression pour chaque type de polluant
    """
    try:
        # Identifier les colonnes de pollution
        colonnes_pollution = [col for col in df.columns if col.startswith('Moyenne_')]

        # Créer une figure avec un sous-graphique par polluant
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()  # Aplatir pour faciliter l'itération

        # Stocker les résultats statistiques
        resultats_correlations = {}

        for i, polluant in enumerate(colonnes_pollution):
            # Filtrer les données pour n'avoir que les lignes complètes
            df_filtered = df.dropna(subset=['Revenu_moyen_par_foyer', polluant])

            if len(df_filtered) < 3:
                print(f"ERREUR: Pas assez de données pour {polluant}")
                continue

            # Calculer la corrélation
            try:
                corr, p_value = stats.pearsonr(
                    df_filtered['Revenu_moyen_par_foyer'].values,
                    df_filtered[polluant].values
                )
                resultats_correlations[polluant] = {
                    'correlation': corr,
                    'p_value': p_value
                }
            except Exception as e:
                print(f"Erreur lors du calcul de la corrélation pour {polluant}: {e}")
                continue

            # Créer le nuage de points avec régression
            sns.regplot(
                x='Revenu_moyen_par_foyer',
                y=polluant,
                data=df_filtered,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'},
                ax=axes[i]
            )

            # Ajouter le titre et les labels
            axes[i].set_title(
                f'{polluant}\nCorrélation: {corr:.3f} (p-value: {p_value:.3f})',
                fontsize=12)
            axes[i].set_xlabel('Revenu moyen par foyer (€)', fontsize=10)
            axes[i].set_ylabel(f'Concentration de {polluant}', fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)

            # Annoter quelques communes significatives
            try:
                if 'Libellé de la commune' in df_filtered.columns:
                    # Identifier quelques communes intéressantes
                    # Communes avec les valeurs de pollution les plus élevées
                    df_sorted_pollution = df_filtered.sort_values(polluant, ascending=False)
                    communes_pollution_elevee = df_sorted_pollution.head(3)

                    # Communes avec les revenus les plus élevés
                    df_sorted_revenu = df_filtered.sort_values('Revenu_moyen_par_foyer', ascending=False)
                    communes_riches = df_sorted_revenu.head(3)

                    # Combiner et supprimer les doublons
                    communes_a_annoter = pd.concat([communes_pollution_elevee, communes_riches]).drop_duplicates()

                    for _, commune in communes_a_annoter.iterrows():
                        axes[i].annotate(
                            commune['Libellé de la commune'],
                            xy=(commune['Revenu_moyen_par_foyer'], commune[polluant]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                        )
            except Exception as e:
                print(f"Erreur lors de l'annotation des communes pour {polluant}: {e}")

        plt.tight_layout()

        if fichier_sortie:
            plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {fichier_sortie}")
        else:
            plt.show()

        plt.close()

        return resultats_correlations
    except Exception as e:
        print(f"Erreur lors de la création des nuages de points: {e}")
        traceback.print_exc()
        return None


def visualiser_correlation_pollution_composite(df, fichier_sortie=None):
    """
    Visualiser la corrélation entre revenu et indice de pollution composite
    """
    try:
        # Filtrer les données pour n'avoir que les lignes complètes
        df_filtered = df.dropna(subset=['Revenu_moyen_par_foyer', 'Indice_pollution_composite'])

        if len(df_filtered) < 3:
            print("ERREUR: Pas assez de données complètes pour calculer la corrélation")
            return None

        plt.figure(figsize=(12, 8))

        # Créer un nuage de points avec régression
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
        except Exception as e:
            print(f"Erreur lors du calcul de la corrélation: {e}")
            corr, p_value = None, None

        # Titre avec statistiques de corrélation
        plt.title(
            f'Relation entre niveau de revenu et indice de pollution composite\n'
            f'Coefficient de corrélation: {corr:.3f} (p-value: {p_value:.3f})' if corr is not None else
            'Relation entre niveau de revenu et indice de pollution composite',
            fontsize=14)

        plt.xlabel('Revenu moyen par foyer (€)', fontsize=12)
        plt.ylabel('Indice de pollution composite', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Annoter les communes significatives
        try:
            if 'Libellé de la commune' in df_filtered.columns:
                # Identifier quelques communes intéressantes
                df_sorted_pollution = df_filtered.sort_values('Indice_pollution_composite', ascending=False)
                communes_pollution_elevee = df_sorted_pollution.head(3)

                df_sorted_revenu = df_filtered.sort_values('Revenu_moyen_par_foyer', ascending=False)
                communes_riches = df_sorted_revenu.head(3)

                # Combiner et supprimer les doublons
                communes_a_annoter = pd.concat([communes_pollution_elevee, communes_riches]).drop_duplicates()

                for _, commune in communes_a_annoter.iterrows():
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

        return {
            'correlation': corr,
            'p_value': p_value
        }

    except Exception as e:
        print(f"Erreur lors de la création du nuage de points: {e}")
        traceback.print_exc()
        return None


def visualiser_heatmap_correlations_par_polluant(df, fichier_sortie=None):
    """
    Créer une heatmap des corrélations entre indicateurs économiques et polluants
    """
    try:
        # Identifier les colonnes de pollution, y compris l'indice composite
        colonnes_pollution = [col for col in df.columns if col.startswith('Moyenne_')]
        colonnes_pollution.append('Indice_pollution_composite')

        # Colonnes économiques
        colonnes_economiques = ['Revenu_moyen_par_foyer', 'Pourcentage_foyers_imposes']

        # Colonnes à analyser
        colonnes_analyse = colonnes_economiques + colonnes_pollution

        # Filtrer les données pour n'avoir que les lignes complètes
        df_filtered = df.dropna(subset=colonnes_analyse)

        plt.figure(figsize=(12, 10))

        # Calculer la matrice de corrélation
        corr_matrix = df_filtered[colonnes_analyse].corr()

        # Personnaliser l'ordre des colonnes pour mettre les indicateurs économiques à gauche
        ordre_colonnes = colonnes_economiques + colonnes_pollution
        corr_matrix = corr_matrix.reindex(index=ordre_colonnes, columns=ordre_colonnes)

        # Masquer la diagonale supérieure
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
            square=False,
            linewidths=.5,
            fmt=".2f"
        )

        plt.title('Matrice de corrélation : Indicateurs économiques vs Pollution', fontsize=14)
        plt.tight_layout()

        if fichier_sortie:
            plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée dans {fichier_sortie}")
        else:
            plt.show()

        plt.close()

        return corr_matrix

    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {e}")
        traceback.print_exc()
        return None


def create_advanced_visualizations(fichier_donnees, dossier_sortie='visualisations'):
    """
    Créer des visualisations avancées à partir d'un fichier de données
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)

    # Charger les données
    try:
        df = pd.read_csv(fichier_donnees)
        print(f"Données chargées: {df.shape[0]} communes, {df.shape[1]} colonnes")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        traceback.print_exc()
        return

    # Créer un rapport détaillé
    try:
        with open(os.path.join(dossier_sortie, 'rapport_analyse_avance.txt'), 'w') as f:
            f.write("RAPPORT D'ANALYSE DÉTAILLÉ - LIEN ENTRE RICHESSE ET POLLUTION\n")
            f.write("=" * 70 + "\n\n")

            # 1. Corrélations par type de polluant
            f.write("1. CORRÉLATIONS PAR TYPE DE POLLUANT\n")
            f.write("-" * 40 + "\n")

            # Nuage de points par polluant
            correlations_par_polluant = visualiser_correlation_par_polluant(
                df,
                os.path.join(dossier_sortie, 'correlation_par_polluant.png')
            )

            if correlations_par_polluant:
                for polluant, resultats in correlations_par_polluant.items():
                    f.write(f"{polluant}:\n")
                    f.write(f"  * Coefficient de corrélation avec revenu: {resultats['correlation']:.3f}\n")
                    f.write(f"  * P-value: {resultats['p_value']:.4f}\n")

                    # Interprétation de la corrélation
                    if abs(resultats['correlation']) < 0.2:
                        interpretation = "Très faible corrélation ou absence de corrélation"
                    elif abs(resultats['correlation']) < 0.4:
                        interpretation = "Faible corrélation"
                    elif abs(resultats['correlation']) < 0.6:
                        interpretation = "Corrélation modérée"
                    elif abs(resultats['correlation']) < 0.8:
                        interpretation = "Forte corrélation"
                    else:
                        interpretation = "Très forte corrélation"

                    if resultats['correlation'] < 0:
                        sens = "négative"
                        explication = "les communes plus riches tendent à avoir moins de ce polluant"
                    else:
                        sens = "positive"
                        explication = "les communes plus riches tendent à avoir plus de ce polluant"

                    f.write(f"  * Interprétation: {interpretation} {sens} - {explication}\n")

                    # Significativité
                    if resultats['p_value'] < 0.01:
                        f.write("  * Statistiquement très significatif (p<0.01)\n")
                    elif resultats['p_value'] < 0.05:
                        f.write("  * Statistiquement significatif (p<0.05)\n")
                    elif resultats['p_value'] < 0.1:
                        f.write("  * Marginalement significatif (p<0.1)\n")
                    else:
                        f.write("  * Non significatif (p>0.1)\n")
                    f.write("\n")

            # 2. Heatmap des corrélations par polluant
            visualiser_heatmap_correlations_par_polluant(
                df,
                os.path.join(dossier_sortie, 'heatmap_correlations_par_polluant.png')
            )

            # 3. Corrélation de l'indice de pollution composite
            f.write("2. CORRÉLATION DE L'INDICE DE POLLUTION COMPOSITE\n")
            f.write("-" * 50 + "\n")

            # Nuage de points pour l'indice composite
            corr_composite = visualiser_correlation_pollution_composite(
                df,
                os.path.join(dossier_sortie, 'correlation_pollution_composite.png')
            )

            if corr_composite:
                f.write(f"Coefficient de corrélation de l'indice composite: {corr_composite['correlation']:.3f}\n")
                f.write(f"P-value: {corr_composite['p_value']:.4f}\n")

                # Interprétation de la corrélation
                if abs(corr_composite['correlation']) < 0.2:
                    interpretation = "Très faible corrélation ou absence de corrélation"
                elif abs(corr_composite['correlation']) < 0.4:
                    interpretation = "Faible corrélation"
                elif abs(corr_composite['correlation']) < 0.6:
                    interpretation = "Corrélation modérée"
                elif abs(corr_composite['correlation']) < 0.8:
                    interpretation = "Forte corrélation"
                else:
                    interpretation = "Très forte corrélation"

                if corr_composite['correlation'] < 0:
                    sens = "négative"
                    explication = "les communes plus riches tendent à avoir moins de pollution globale"
                else:
                    sens = "positive"
                    explication = "les communes plus riches tendent à avoir plus de pollution globale"

                f.write(f"Interprétation : {interpretation} {sens} - {explication}\n")

                # Significativité
                if corr_composite['p_value'] < 0.01:
                    f.write("Statistiquement très significatif (p<0.01)\n")
                elif corr_composite['p_value'] < 0.05:
                    f.write("Statistiquement significatif (p<0.05)\n")
                elif corr_composite['p_value'] < 0.1:
                    f.write("Marginalement significatif (p<0.1)\n")
                else:
                    f.write("Non significatif (p>0.1)\n")

        print(f"\nRapport d'analyse détaillé créé: {os.path.join(dossier_sortie, 'rapport_analyse_avance.txt')}")

    except Exception as e:
        print(f"Erreur lors de la création du rapport détaillé: {e}")
        traceback.print_exc()


def main():
    """
    Fonction principale d'exécution pour les visualisations avancées
    """
    import argparse

    parser = argparse.ArgumentParser(description="Créer des visualisations avancées pour l'analyse pollution-revenus")
    parser.add_argument('--data', type=str, required=True, help='Chemin du fichier CSV avec les données analysées')
    parser.add_argument('--output', type=str, default='visualisations_avancees',
                        help='Dossier de sortie pour les visualisations')
    parser.add_argument('--annee', type=str, help='Année des données (pour le nom des fichiers)')

    args = parser.parse_args()

    # Créer le dossier de sortie avec l'année si spécifiée
    dossier_sortie = args.output
    if args.annee:
        dossier_sortie = f"{args.output}_{args.annee}"

    create_advanced_visualizations(args.data, dossier_sortie)


if __name__ == "__main__":
    main()