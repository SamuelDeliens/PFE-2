import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


# Fonction pour charger les données
def load_data(gdp_file, air_quality_file):
    """Charge les données de PIB et de qualité d'air à partir des fichiers JSON"""
    with open(gdp_file, 'r', encoding='utf-8') as f:
        gdp_data = json.load(f)

    with open(air_quality_file, 'r', encoding='utf-8') as f:
        air_quality_data = json.load(f)

    # Retirer la Turquie des données d'air quality si elle existe
    if "Turkey" in air_quality_data:
        print("Exclusion de la Turquie des données de qualité d'air")
        del air_quality_data["Turkey"]

    return gdp_data, air_quality_data


# Fonction pour calculer les moyennes annuelles de pollution
def calculate_yearly_pollution(air_quality_data):
    """Calcule les moyennes annuelles de pollution pour chaque pays et polluant"""
    yearly_pollution = {}
    all_pollutants = set()

    # Première passe pour découvrir tous les polluants
    for country, dates in air_quality_data.items():
        for date_key, pollutants in dates.items():
            for pollutant in pollutants:
                all_pollutants.add(pollutant)

    for country, dates in air_quality_data.items():
        # Ignorer la Turquie
        if country == "Turkey":
            continue

        yearly_pollution[country] = {}

        for date_key, pollutants in dates.items():
            year = date_key.split('-')[0]

            if year not in yearly_pollution[country]:
                yearly_pollution[country][year] = {}
                for pollutant in all_pollutants:
                    yearly_pollution[country][year][pollutant] = {'sum': 0, 'count': 0}

            for pollutant, data in pollutants.items():
                # Vérifier la présence de la clé 'value' ou utiliser une alternative
                pollution_value = None
                if 'value' in data:
                    pollution_value = data['value']
                elif 'avg' in data:
                    pollution_value = data['avg']
                elif 'median' in data:
                    pollution_value = data['median']

                if pollution_value is not None:
                    yearly_pollution[country][year][pollutant]['sum'] += pollution_value
                    yearly_pollution[country][year][pollutant]['count'] += 1

        # Calculer les moyennes
        for year in yearly_pollution[country]:
            for pollutant in list(yearly_pollution[country][year].keys()):
                if yearly_pollution[country][year][pollutant]['count'] > 0:
                    yearly_pollution[country][year][pollutant] = (
                            yearly_pollution[country][year][pollutant]['sum'] /
                            yearly_pollution[country][year][pollutant]['count']
                    )
                else:
                    yearly_pollution[country][year][pollutant] = None

    return yearly_pollution, all_pollutants


# Fonction pour créer un dictionnaire des PIB par pays et par année
def create_gdp_dict(gdp_data):
    """Crée un dictionnaire des PIB par pays et par année"""
    gdp_dict = {}

    for entry in gdp_data:
        country_name = entry.get('Country Name')
        # Ignorer la Turquie
        if country_name and country_name != "Turkey":
            gdp_dict[country_name] = entry.get('GDP', {})

    return gdp_dict


# Fonction pour créer des paires PIB-pollution
def create_gdp_pollution_pairs(gdp_dict, yearly_pollution, all_pollutants):
    """Crée des paires (PIB, pollution) pour chaque pays, année et polluant"""
    pairs = {}

    # Initialiser les listes pour chaque polluant
    for pollutant in all_pollutants:
        pairs[pollutant] = []

    # Pour chaque pays dans les données de pollution
    for country, years in yearly_pollution.items():
        # Ignorer la Turquie
        if country == "Turkey":
            continue

        # Si nous avons aussi des données de PIB pour ce pays
        if country in gdp_dict:
            for year, pollutants in years.items():
                # Si nous avons des données de PIB pour cette année
                if year in gdp_dict[country]:
                    gdp = gdp_dict[country][year]

                    # Pour chaque type de polluant
                    for pollutant, value in pollutants.items():
                        # Si nous avons une valeur de pollution valide
                        if value is not None:
                            pairs[pollutant].append({
                                'country': country,
                                'year': year,
                                'gdp': float(gdp),
                                'pollution': float(value)
                            })

    return pairs


# Fonction pour calculer la corrélation
def calculate_correlations(pairs):
    """Calcule la corrélation entre le PIB et la pollution pour chaque type de polluant"""
    correlations = {}

    for pollutant, data in pairs.items():
        if len(data) > 1:
            gdp_values = [point['gdp'] for point in data]
            pollution_values = [point['pollution'] for point in data]

            try:
                correlation, p_value = pearsonr(gdp_values, pollution_values)
                r_squared = correlation ** 2

                correlations[pollutant] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'r_squared': r_squared,
                    'n': len(data)
                }
            except Exception as e:
                print(f"Erreur lors du calcul de la corrélation pour {pollutant}: {e}")
                correlations[pollutant] = {
                    'correlation': None,
                    'p_value': None,
                    'r_squared': None,
                    'n': len(data)
                }
        else:
            correlations[pollutant] = {
                'correlation': None,
                'p_value': None,
                'r_squared': None,
                'n': len(data)
            }

    return correlations


# Fonction pour créer un DataFrame à partir des paires
def create_dataframe(pairs):
    """Convertit les paires en DataFrame pour faciliter la visualisation"""
    dfs = {}
    all_data = []

    for pollutant, data in pairs.items():
        if data:
            # Créer un DataFrame excluant la Turquie
            df = pd.DataFrame([d for d in data if d['country'] != "Turkey"])
            df['pollutant'] = pollutant
            dfs[pollutant] = df
            all_data.extend([d for d in data if d['country'] != "Turkey"])

    # Créer un DataFrame global avec tous les polluants, excluant la Turquie
    if all_data:
        global_df = pd.DataFrame(all_data)
    else:
        global_df = pd.DataFrame()

    return dfs, global_df


# Fonction pour visualiser tous les nuages de points sur une seule figure
def plot_all_scatterplots(dfs, correlations, output_file='gdp_pollution_all.png'):
    """Crée une figure avec tous les nuages de points"""
    # Compter le nombre de polluants avec des données
    valid_pollutants = [p for p, df in dfs.items() if not df.empty and len(df) > 2]
    n = len(valid_pollutants)

    if n == 0:
        print("Aucun polluant avec suffisamment de données pour la visualisation.")
        return

    # Calculer la disposition de la grille
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    # Créer une figure avec une disposition en grille
    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig)

    # Créer un nuage de points pour chaque polluant
    for i, pollutant in enumerate(valid_pollutants):
        row = i // cols
        col = i % cols

        ax = fig.add_subplot(gs[row, col])
        df = dfs[pollutant]

        # Créer le nuage de points
        scatter = ax.scatter(df['gdp'], df['pollution'], alpha=0.7)

        # Ajouter une régression linéaire
        if len(df) > 2:
            m, b = np.polyfit(df['gdp'], df['pollution'], 1)
            x_range = np.linspace(df['gdp'].min(), df['gdp'].max(), 100)
            ax.plot(x_range, m * x_range + b, 'r-')

        # Ajouter les étiquettes et le titre
        ax.set_xlabel('PIB par habitant ($)')
        ax.set_ylabel(f'Pollution ({pollutant})')
        ax.set_title(f'PIB vs {pollutant}')

        # Ajouter une annotation pour les pays (pour les 5 valeurs les plus élevées de pollution)
        if len(df) > 5:
            top_points = df.nlargest(5, 'pollution')
            for _, point in top_points.iterrows():
                ax.annotate(
                    point['country'],
                    (point['gdp'], point['pollution']),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left'
                )

        # Ajouter les informations de corrélation
        if pollutant in correlations and correlations[pollutant]['correlation'] is not None:
            corr = correlations[pollutant]['correlation']
            r2 = correlations[pollutant]['r_squared']
            n = correlations[pollutant]['n']
            ax.text(0.05, 0.95,
                    f'Corr: {corr:.3f}\nR²: {r2:.3f}\nN: {n}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graphique global enregistré sous '{output_file}'")


# Fonction pour créer une matrice de nuages de points
def plot_scatterplot_matrix(dfs, output_file='gdp_pollution_matrix.png'):
    """Crée une matrice de nuages de points pour visualiser la relation PIB-pollution pour plusieurs années"""
    # Préparer un DataFrame combiné pour tous les types de polluants
    combined_df = pd.DataFrame()

    for pollutant, df in dfs.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['pollutant'] = pollutant
            combined_df = pd.concat([combined_df, df_copy])

    if combined_df.empty:
        print("Pas de données suffisantes pour créer une matrice de nuages de points.")
        return

    # S'assurer que la Turquie est exclue
    combined_df = combined_df[combined_df['country'] != "Turkey"]

    # Créer une palette de couleurs pour distinguer les polluants
    pollutants = combined_df['pollutant'].unique()
    colors = sns.color_palette("husl", len(pollutants))
    pollutant_colors = dict(zip(pollutants, colors))

    # Créer la matrice de nuages de points par année
    years = sorted(combined_df['year'].unique())

    if len(years) < 2:
        print("Pas assez d'années différentes pour créer une matrice.")
        return

    # Limiter à 9 années pour la lisibilité
    if len(years) > 9:
        years = sorted(years)[-9:]

    # Calculer la disposition de la grille
    cols = min(3, len(years))
    rows = (len(years) + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig)

    for i, year in enumerate(years):
        row = i // cols
        col = i % cols

        ax = fig.add_subplot(gs[row, col])
        year_df = combined_df[combined_df['year'] == year]

        for pollutant in pollutants:
            pollutant_df = year_df[year_df['pollutant'] == pollutant]
            if not pollutant_df.empty and len(pollutant_df) > 1:
                ax.scatter(
                    pollutant_df['gdp'],
                    pollutant_df['pollution'],
                    color=pollutant_colors[pollutant],
                    alpha=0.7,
                    label=pollutant
                )

                # Ajouter des lignes de tendance si suffisamment de points
                if len(pollutant_df) > 2:
                    try:
                        m, b = np.polyfit(pollutant_df['gdp'], pollutant_df['pollution'], 1)
                        x_range = np.linspace(pollutant_df['gdp'].min(), pollutant_df['gdp'].max(), 100)
                        ax.plot(x_range, m * x_range + b, color=pollutant_colors[pollutant], linestyle='--', alpha=0.5)
                    except Exception as e:
                        print(f"Erreur lors de l'ajout de la ligne de tendance pour {pollutant} en {year}: {e}")

        ax.set_xlabel('PIB par habitant ($)')
        ax.set_ylabel('Niveau de pollution')
        ax.set_title(f'Année {year}')

        # Ajouter la légende seulement au premier graphique
        if i == 0:
            ax.legend(title="Polluants")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matrice de nuages de points enregistrée sous '{output_file}'")


# Fonction pour visualiser les tendances temporelles
def plot_time_trends(dfs, output_file='gdp_pollution_trends.png'):
    """Crée un graphique montrant la relation PIB-pollution au fil du temps pour différents pays"""
    # Limiter aux polluants ayant suffisamment de données
    valid_pollutants = [p for p, df in dfs.items() if not df.empty and len(df) > 5]

    if not valid_pollutants:
        print("Pas de données suffisantes pour analyser les tendances temporelles.")
        return

    # Sélectionner les pays qui ont des données pour plusieurs années
    all_countries = set()
    for pollutant in valid_pollutants:
        df = dfs[pollutant]
        # S'assurer que la Turquie est exclue
        df = df[df['country'] != "Turkey"]
        country_counts = df['country'].value_counts()
        countries_with_multiple_years = country_counts[country_counts > 1].index.tolist()
        all_countries.update(countries_with_multiple_years)

    # Limiter à 9 pays pour la lisibilité
    top_countries = list(all_countries)[:9] if len(all_countries) > 9 else list(all_countries)

    if not top_countries:
        print("Pas de pays avec plusieurs années de données.")
        return

    # Créer une figure pour les tendances temporelles
    cols = min(3, len(top_countries))
    rows = (len(top_countries) + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig)

    # Créer une palette de couleurs pour les polluants
    colors = sns.color_palette("husl", len(valid_pollutants))
    pollutant_colors = dict(zip(valid_pollutants, colors))

    for i, country in enumerate(top_countries):
        row = i // cols
        col = i % cols

        ax = fig.add_subplot(gs[row, col])

        # Tracer les données pour chaque polluant
        for pollutant in valid_pollutants:
            df = dfs[pollutant]
            # S'assurer que la Turquie est exclue
            df = df[df['country'] != "Turkey"]
            country_df = df[df['country'] == country]

            if not country_df.empty and len(country_df) > 1:
                # Trier par année
                country_df = country_df.sort_values('year')
                years = country_df['year'].astype(str).tolist()
                gdp_values = country_df['gdp'].tolist()
                pollution_values = country_df['pollution'].tolist()

                # Créer des lignes pour relier les points
                for j in range(len(years) - 1):
                    ax.plot(
                        [gdp_values[j], gdp_values[j + 1]],
                        [pollution_values[j], pollution_values[j + 1]],
                        color=pollutant_colors[pollutant],
                        alpha=0.7,
                        marker='o'
                    )

                # Ajouter les années comme annotations
                for j, (year, gdp, poll) in enumerate(zip(years, gdp_values, pollution_values)):
                    ax.annotate(
                        year,
                        (gdp, poll),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center'
                    )

        ax.set_xlabel('PIB par habitant ($)')
        ax.set_ylabel('Niveau de pollution')
        ax.set_title(f'Tendances pour {country}')

        # Ajouter la légende seulement au premier graphique
        if i == 0:
            legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=pollutant)
                               for pollutant, color in pollutant_colors.items()]
            ax.legend(handles=legend_elements, title="Polluants")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graphique des tendances temporelles enregistré sous '{output_file}'")


# Fonction pour créer une visualisation combinée
def create_combined_visualization(dfs, correlations, output_file='combined_analysis.png'):
    """Crée une visualisation combinée pour tous les polluants"""
    # Préparer un DataFrame combiné
    combined_df = pd.DataFrame()
    for pollutant, df in dfs.items():
        if not df.empty and len(df) > 2:
            df_copy = df.copy()
            df_copy['pollutant'] = pollutant
            combined_df = pd.concat([combined_df, df_copy])

    if combined_df.empty:
        print("Pas de données suffisantes pour la visualisation combinée.")
        return

    # S'assurer que la Turquie est exclue
    combined_df = combined_df[combined_df['country'] != "Turkey"]

    # Créer une figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # 1. Nuage de points global
    sns.scatterplot(
        data=combined_df,
        x='gdp',
        y='pollution',
        hue='pollutant',
        alpha=0.7,
        ax=ax1
    )

    ax1.set_xlabel('PIB par habitant ($)')
    ax1.set_ylabel('Niveau de pollution')
    ax1.set_title('Relation globale entre PIB et pollution')
    ax1.legend(title="Polluants", loc="upper right")

    # 2. Graphique à barres des corrélations
    pollutants = []
    corr_values = []
    errors = []

    for pollutant, stats in correlations.items():
        if stats['correlation'] is not None and pollutant in dfs and not dfs[pollutant].empty:
            pollutants.append(pollutant)
            corr_values.append(stats['correlation'])
            # Calculer l'erreur standard (approximative) pour la barre d'erreur
            n = stats['n']
            if n > 3:  # Au moins 4 points pour une estimation d'erreur
                error = (1 - stats['correlation'] ** 2) / np.sqrt(n - 2)
                errors.append(error)
            else:
                errors.append(0)

    # Trier par valeur absolue de corrélation
    sorted_indices = np.argsort(np.abs(corr_values))[::-1]
    pollutants = [pollutants[i] for i in sorted_indices]
    corr_values = [corr_values[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]

    bars = ax2.barh(pollutants, corr_values, xerr=errors, alpha=0.7)

    # Colorer les barres selon la direction de la corrélation
    for i, bar in enumerate(bars):
        if corr_values[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Coefficient de corrélation')
    ax2.set_title('Force de la relation PIB-pollution par polluant')

    # Ajouter des lignes pour marquer les seuils de corrélation
    ax2.axvline(x=0.7, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=-0.7, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0.3, color='black', linestyle=':', alpha=0.3)
    ax2.axvline(x=-0.3, color='black', linestyle=':', alpha=0.3)

    ax2.text(0.7, len(pollutants) - 0.5, 'Forte +', ha='left', va='center', alpha=0.7)
    ax2.text(-0.7, len(pollutants) - 0.5, 'Forte -', ha='right', va='center', alpha=0.7)

    # Ajouter le nombre d'observations pour chaque polluant
    for i, pollutant in enumerate(pollutants):
        if pollutant in correlations:
            n = correlations[pollutant]['n']
            ax2.text(
                max(0.05, abs(corr_values[i]) + errors[i] + 0.05) * np.sign(corr_values[i]),
                i,
                f'n={n}',
                va='center'
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualisation combinée enregistrée sous '{output_file}'")


# Fonction principale
def main(gdp_file, air_quality_file, output_dir='visualisations'):
    """Fonction principale pour générer toutes les visualisations"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Chargement des données (en excluant la Turquie)...")
    # Charger les données
    gdp_data, air_quality_data = load_data(gdp_file, air_quality_file)

    print("Calcul des moyennes annuelles de pollution...")
    # Calculer les moyennes annuelles de pollution
    yearly_pollution, all_pollutants = calculate_yearly_pollution(air_quality_data)

    print("Création du dictionnaire des PIB...")
    # Créer un dictionnaire des PIB
    gdp_dict = create_gdp_dict(gdp_data)

    print("Création des paires PIB-pollution...")
    # Créer des paires PIB-pollution
    pairs = create_gdp_pollution_pairs(gdp_dict, yearly_pollution, all_pollutants)

    print("Calcul des corrélations...")
    # Calculer les corrélations
    correlations = calculate_correlations(pairs)

    print("Création des DataFrames...")
    # Créer des DataFrames
    dfs, global_df = create_dataframe(pairs)

    print("Création des visualisations...")
    # 1. Nuages de points individuels pour chaque polluant sur une seule figure
    plot_all_scatterplots(dfs, correlations, os.path.join(output_dir, 'gdp_pollution_all.png'))

    # 2. Matrice de nuages de points par année
    plot_scatterplot_matrix(dfs, os.path.join(output_dir, 'gdp_pollution_matrix.png'))

    # 3. Tendances temporelles pour les pays ayant plusieurs années de données
    plot_time_trends(dfs, os.path.join(output_dir, 'gdp_pollution_trends.png'))

    # 4. Visualisation combinée pour une analyse globale
    create_combined_visualization(dfs, correlations, os.path.join(output_dir, 'combined_analysis.png'))

    # Afficher les résultats de corrélation
    print("\nRésultats de corrélation (sans la Turquie):")
    for pollutant, stats in correlations.items():
        if stats['correlation'] is not None:
            print(f"\n{pollutant}:")
            print(f"  Corrélation: {stats['correlation']:.3f}")
            print(f"  R²: {stats['r_squared']:.3f}")
            print(f"  Valeur p: {stats['p_value']:.5f}")
            print(f"  Nombre d'observations: {stats['n']}")

            # Interprétation
            if abs(stats['correlation']) > 0.7:
                strength = "forte"
            elif abs(stats['correlation']) > 0.3:
                strength = "modérée"
            else:
                strength = "faible"

            direction = "positive" if stats['correlation'] > 0 else "négative"

            print(f"  Interprétation: Corrélation {strength} {direction} entre le PIB et la pollution {pollutant}.")
        else:
            print(f"\n{pollutant}:")
            print(f"  Données insuffisantes pour calculer une corrélation (n={stats['n']}).")

    print(f"\nToutes les visualisations ont été enregistrées dans le répertoire '{output_dir}'")

    # Retourner les résultats pour une éventuelle utilisation ultérieure
    return dfs, correlations


if __name__ == "__main__":
    # Chemins des fichiers
    gdp_file = "./data/revenus-fiscaux/PIB/PIB_2016_2023.json"
    air_quality_file = "./data/air-quality/pollutant_data_completed.json"

    try:
        # Exécuter l'analyse en excluant la Turquie
        main(gdp_file, air_quality_file)
    except Exception as e:
        print(f"Erreur lors de l'exécution du programme: {e}")
