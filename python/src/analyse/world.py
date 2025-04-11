import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Set, Optional
from typing import Dict, List, Tuple, Any, Set, Optional

class NumpyEncoder(json.JSONEncoder):
    """
    Encodeur JSON personnalisé qui gère les types NumPy
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # Gérer les valeurs NaN/None
            return None
        return super(NumpyEncoder, self).default(obj)
def convert_numpy_types(obj):
    """
    Convertit récursivement les types NumPy en types Python natifs pour la sérialisation JSON

    Args:
        obj: Objet à convertir

    Returns:
        Objet avec les types NumPy convertis
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif pd.isna(obj):  # Gérer les valeurs NaN/None
        return None
    return obj
def save_results_to_json(results, filename):
    """
    Sauvegarde les résultats en JSON en gérant correctement les types NumPy

    Args:
        results: Dictionnaire de résultats
        filename: Nom du fichier de destination
    """
    # Convertir les types NumPy en types Python natifs
    converted_results = convert_numpy_types(results)

    # Sauvegarder en JSON
    with open(filename, 'w') as f:
        json.dump(converted_results, f, indent=2, cls=NumpyEncoder)

    print(f"Résultats sauvegardés dans {filename}")


def analyze_global_pollution_wealth_relation(sensor_data: Dict, gdp_data: List[Dict]) -> Dict:
    """
    Fonction principale pour analyser la relation entre les multiples polluants et richesse

    Args:
        sensor_data: Données des capteurs par pays, par date et par type de polluant
        gdp_data: Données PIB par pays et par année

    Returns:
        Dictionnaire contenant les résultats de l'analyse
    """
    # Structure pour stocker les résultats
    results = {
        "countries_data": {},  # Résultats par pays
        "global_stats": {},  # Statistiques globales
        "raw_data": [],  # Données brutes pour analyses ultérieures
        "pollutant_stats": {}  # Statistiques par type de polluant
    }

    # Identifier les types de polluants disponibles
    pollutant_types = identify_pollutant_types(sensor_data)
    print(f"Types de polluants identifiés: {', '.join(pollutant_types)}")

    # Extraction des années/mois disponibles dans les données
    available_periods = extract_available_periods(sensor_data, gdp_data)
    print(f"Périodes disponibles pour l'analyse: {', '.join(available_periods)}")

    # Calcul des valeurs min/max globales pour chaque type de polluant (pour la normalisation)
    min_max_values = calculate_global_min_max_by_pollutant(
        sensor_data, gdp_data, available_periods, pollutant_types
    )

    for pollutant in pollutant_types:
        min_val = min_max_values[pollutant]["min_pollution"]
        max_val = min_max_values[pollutant]["max_pollution"]
        print(f"Bornes de normalisation - {pollutant}: [{min_val:.2f}, {max_val:.2f}]")

    print(
        f"Bornes de normalisation - PIB: [{min_max_values['gdp']['min_gdp']:.2f}, {min_max_values['gdp']['max_gdp']:.2f}]")

    # Pour chaque pays présent dans les deux jeux de données
    common_countries = find_common_countries(sensor_data, gdp_data)
    print(f"Nombre de pays avec données complètes: {len(common_countries)}")

    # Traitement pour chaque pays
    for country in common_countries:
        country_gdp = find_gdp_for_country(gdp_data, country)
        country_data = process_country_data(
            country,
            sensor_data[country],
            country_gdp,
            available_periods,
            min_max_values,
            pollutant_types
        )

        # Stocker les résultats pour ce pays
        results["countries_data"][country] = country_data

        # Ajouter les données aux données brutes pour les statistiques globales
        for period in country_data["indices"]:
            for pollutant in country_data["indices"][period]:
                for index_name, value in country_data["indices"][period][pollutant].items():
                    results["raw_data"].append({
                        "country": country,
                        "period": period,
                        "pollutant": pollutant,
                        "index_name": index_name,
                        "value": value
                    })

    # Calcul des statistiques globales
    results["global_stats"] = calculate_global_statistics(results["raw_data"])

    # Calcul des statistiques par type de polluant
    results["pollutant_stats"] = calculate_pollutant_statistics(results["raw_data"])

    return results


def identify_pollutant_types(sensor_data: Dict) -> List[str]:
    """
    Identifie tous les types de polluants disponibles dans les données
    """
    pollutant_types = set()

    for country, periods in sensor_data.items():
        for period, pollutants in periods.items():
            for pollutant in pollutants:
                pollutant_types.add(pollutant)

    return sorted(list(pollutant_types))


def extract_available_periods(sensor_data: Dict, gdp_data: List[Dict]) -> List[str]:
    """
    Extrait les périodes (année-mois) disponibles dans les données des capteurs
    et vérifie la correspondance avec les années des données PIB
    """
    sensor_periods = set()
    for country, periods in sensor_data.items():
        for period in periods:
            sensor_periods.add(period)

    # Extraire les années des périodes (YYYY-MM -> YYYY)
    sensor_years = {period.split('-')[0] for period in sensor_periods}

    gdp_years = set()
    for country_data in gdp_data:
        if "GDP" in country_data:
            for year in country_data["GDP"]:
                gdp_years.add(year)

    # Vérifier quelles années sont disponibles dans les deux jeux de données
    valid_years = sorted(list(sensor_years.intersection(gdp_years)))

    # Filtrer les périodes dont l'année est valide
    valid_periods = sorted([period for period in sensor_periods
                            if period.split('-')[0] in valid_years])

    return valid_periods


def calculate_global_min_max_by_pollutant(
        sensor_data: Dict,
        gdp_data: List[Dict],
        periods: List[str],
        pollutant_types: List[str]
) -> Dict:
    """
    Calcule les valeurs minimales et maximales globales pour chaque type de polluant
    et pour le PIB (pour la normalisation)
    """
    result = {
        'gdp': {
            'min_gdp': float('inf'),
            'max_gdp': float('-inf')
        }
    }

    # Initialiser les min/max pour chaque type de polluant
    for pollutant in pollutant_types:
        result[pollutant] = {
            'min_pollution': float('inf'),
            'max_pollution': float('-inf')
        }

    # Parcourir toutes les données de capteurs
    for country, country_data in sensor_data.items():
        for period, pollutants in country_data.items():
            if period in periods:
                for pollutant_type, pollutant_data in pollutants.items():
                    if pollutant_type in pollutant_types and "value" in pollutant_data:
                        pollution_value = pollutant_data["value"]
                        if pollution_value is not None:
                            result[pollutant_type]['min_pollution'] = min(
                                result[pollutant_type]['min_pollution'], pollution_value
                            )
                            result[pollutant_type]['max_pollution'] = max(
                                result[pollutant_type]['max_pollution'], pollution_value
                            )

    # Parcourir toutes les données PIB
    for country in gdp_data:
        years = {period.split('-')[0] for period in periods}
        for year in years:
            if "GDP" in country and year in country["GDP"]:
                gdp_value = country["GDP"][year]
                if gdp_value is not None:
                    result['gdp']['min_gdp'] = min(result['gdp']['min_gdp'], gdp_value)
                    result['gdp']['max_gdp'] = max(result['gdp']['max_gdp'], gdp_value)

    return result


def find_common_countries(sensor_data: Dict, gdp_data: List[Dict]) -> List[str]:
    """
    Trouve les pays présents dans les deux jeux de données
    """
    sensor_countries = set(sensor_data.keys())
    gdp_countries = {country["Country Name"] for country in gdp_data}

    # Trouver l'intersection
    return list(sensor_countries.intersection(gdp_countries))


def find_gdp_for_country(gdp_data: List[Dict], country_name: str) -> Dict:
    """
    Trouve les données PIB pour un pays donné
    """
    for country in gdp_data:
        if country["Country Name"] == country_name:
            return country
    return None


def process_country_data(
        country_name: str,
        country_sensor_data: Dict,
        country_gdp_data: Dict,
        periods: List[str],
        min_max_values: Dict,
        pollutant_types: List[str]
) -> Dict:
    """
    Traite les données pour un pays spécifique
    """
    result = {
        "name": country_name,
        "period_averages": {},  # Moyennes par période
        "indices": {}  # Indices par période et par polluant
    }

    # Pour chaque période disponible
    for period in periods:
        if period in country_sensor_data:
            year = period.split('-')[0]

            if "GDP" in country_gdp_data and year in country_gdp_data["GDP"]:
                gdp_value = country_gdp_data["GDP"][year]

                # Initialiser les résultats pour cette période
                result["period_averages"][period] = {}
                result["indices"][period] = {}

                # Traiter chaque type de polluant
                for pollutant in pollutant_types:
                    if pollutant in country_sensor_data[period]:
                        # Stocker les moyennes pour ce polluant
                        result["period_averages"][period][pollutant] = extract_pollutant_data(
                            country_sensor_data[period][pollutant]
                        )

                        # Calculer les indices pour ce polluant
                        if "value" in country_sensor_data[period][pollutant]:
                            pollution_value = country_sensor_data[period][pollutant]["value"]

                            # Calculer les indices pour ce polluant
                            result["indices"][period][pollutant] = calculate_indices(
                                pollution_value,
                                gdp_value,
                                min_max_values[pollutant]["min_pollution"],
                                min_max_values[pollutant]["max_pollution"],
                                min_max_values["gdp"]["min_gdp"],
                                min_max_values["gdp"]["max_gdp"]
                            )

    return result


def extract_pollutant_data(pollutant_data: Dict) -> Dict:
    """
    Extrait et organise les données d'un polluant
    """
    result = {}
    for key, value in pollutant_data.items():
        if value is not None:
            result[key] = value

    return result


def calculate_indices(
        pollution_value: float,
        gdp_value: float,
        min_pollution: float,
        max_pollution: float,
        min_gdp: float,
        max_gdp: float
) -> Dict[str, float]:
    """
    Calcule les différents indices de pollution-richesse pour un polluant
    """
    # Vérifier les valeurs nulles ou négatives
    if pollution_value is None or gdp_value is None or pollution_value <= 0 or gdp_value <= 0:
        return {
            "pollution_gdp_ratio": None,
            "gdp_pollution_ratio": None,
            "normalized_ratio": None,
            "env_inequality_index": None,
            "composite_index": None
        }

    # 1. Ratio pollution/PIB (×1000000 pour lisibilité)
    pollution_gdp_ratio = (pollution_value / gdp_value) * 1000000

    # 2. Ratio PIB/pollution (mesure de l'efficacité économique)
    gdp_pollution_ratio = gdp_value / pollution_value

    # 3. Ratio normalisé
    pollution_range = max_pollution - min_pollution
    gdp_range = max_gdp - min_gdp

    if pollution_range > 0 and gdp_range > 0:
        norm_pollution = (pollution_value - min_pollution) / pollution_range
        norm_gdp = (gdp_value - min_gdp) / gdp_range
        normalized_ratio = norm_pollution / norm_gdp if norm_gdp > 0 else float('inf')

        # 4. Indice d'inégalité environnementale (inspiré de l'indice de Gini)
        env_inequality_index = abs(norm_pollution - norm_gdp)

        # 5. Indice composite combinant les précédents
        composite_index = (pollution_gdp_ratio / 1000) + (normalized_ratio * 10) + (env_inequality_index * 5)
    else:
        normalized_ratio = None
        env_inequality_index = None
        composite_index = None

    return {
        "pollution_gdp_ratio": pollution_gdp_ratio,
        "gdp_pollution_ratio": gdp_pollution_ratio,
        "normalized_ratio": normalized_ratio,
        "env_inequality_index": env_inequality_index,
        "composite_index": composite_index
    }


def calculate_global_statistics(raw_data: List[Dict]) -> Dict:
    """
    Calcule les statistiques globales pour tous les pays
    """
    # Convertir en DataFrame pour faciliter l'analyse
    df = pd.DataFrame(raw_data)

    if df.empty:
        return {}

    # Regrouper par indice, période et polluant
    grouped = df.groupby(['index_name', 'period', 'pollutant'])

    stats = {}

    # Pour chaque groupe, calculer les statistiques
    for (index_name, period, pollutant), group in grouped:
        if index_name not in stats:
            stats[index_name] = {}

        if period not in stats[index_name]:
            stats[index_name][period] = {}

        values = group['value'].dropna().values

        if len(values) > 0:
            stats[index_name][period][pollutant] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std_dev': np.std(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'coeff_var': (np.std(values) / np.mean(values) * 100) if np.mean(values) != 0 else None
            }

    return stats


def calculate_pollutant_statistics(raw_data: List[Dict]) -> Dict:
    """
    Calcule des statistiques comparatives entre les différents polluants
    """
    df = pd.DataFrame(raw_data)

    if df.empty:
        return {}

    # Filtrer pour ne garder que les lignes avec des valeurs non nulles
    df = df.dropna(subset=['value'])

    # Résultats par type de polluant
    pollutant_stats = {}

    # 1. Moyenne des indices par polluant
    pollutant_means = df.groupby('pollutant')['value'].mean()

    # 2. Écart-type des indices par polluant
    pollutant_stds = df.groupby('pollutant')['value'].std()

    # 3. Nombre de pays avec données pour chaque polluant
    pollutant_countries = df.groupby('pollutant')['country'].nunique()

    # 4. Pour chaque type d'indice, classement des polluants
    index_rankings = {}
    for index_name in df['index_name'].unique():
        index_data = df[df['index_name'] == index_name]

        # Calculer la moyenne par polluant pour cet indice
        index_means = index_data.groupby('pollutant')['value'].mean().sort_values()

        # Stocker le classement (du plus faible au plus élevé)
        index_rankings[index_name] = index_means.index.tolist()

    # Combiner les résultats
    for pollutant in df['pollutant'].unique():
        pollutant_stats[pollutant] = {
            'mean_across_indices': pollutant_means.get(pollutant, None),
            'std_across_indices': pollutant_stds.get(pollutant, None),
            'countries_count': pollutant_countries.get(pollutant, 0),
            'rankings': {index: rankings.index(pollutant) + 1 if pollutant in rankings else None
                         for index, rankings in index_rankings.items()}
        }

    return pollutant_stats


def generate_report(results: Dict) -> Dict:
    """
    Génère un rapport d'analyse basé sur les résultats
    """
    # Convertir les données brutes en DataFrame pour faciliter l'analyse
    df = pd.DataFrame(results["raw_data"])

    if df.empty:
        return {"error": "Pas de données disponibles pour générer un rapport"}

    report = {
        "summary": {
            "total_countries": len(results["countries_data"]),
            "periods_analyzed": sorted(df['period'].unique().tolist()),
            "pollutants_analyzed": sorted(df['pollutant'].unique().tolist())
        },
        "pollutant_comparisons": {},
        "top_countries_by_pollutant": {},
        "bottom_countries_by_pollutant": {},
        "global_trends": {}
    }

    # Comparaison des différents polluants
    pollutants = report["summary"]["pollutants_analyzed"]
    for pollutant in pollutants:
        if pollutant in results["pollutant_stats"]:
            report["pollutant_comparisons"][pollutant] = results["pollutant_stats"][pollutant]

    # Pour chaque polluant, identifier les pays avec les valeurs les plus élevées/basses
    indices = ["pollution_gdp_ratio", "normalized_ratio", "env_inequality_index", "composite_index"]

    for pollutant in pollutants:
        report["top_countries_by_pollutant"][pollutant] = {}
        report["bottom_countries_by_pollutant"][pollutant] = {}

        for index_name in indices:
            # Filtrer pour ce polluant et cet indice
            filtered_df = df[(df['pollutant'] == pollutant) &
                             (df['index_name'] == index_name)]

            if filtered_df.empty:
                continue

            # Agréger par pays (moyenne sur toutes les périodes)
            country_avg = filtered_df.groupby('country')['value'].mean().reset_index()

            if country_avg.empty:
                continue

            # Top 5
            report["top_countries_by_pollutant"][pollutant][index_name] = (
                country_avg.nlargest(5, 'value')[['country', 'value']].to_dict('records')
            )

            # Bottom 5
            report["bottom_countries_by_pollutant"][pollutant][index_name] = (
                country_avg.nsmallest(5, 'value')[['country', 'value']].to_dict('records')
            )

    # Analyser les tendances globales des indices au fil du temps pour chaque polluant
    for pollutant in pollutants:
        report["global_trends"][pollutant] = {}

        for index_name in indices:
            # Filtrer pour ce polluant et cet indice
            filtered_df = df[(df['pollutant'] == pollutant) &
                             (df['index_name'] == index_name)]

            if filtered_df.empty:
                continue

            # Agréger par période
            period_stats = filtered_df.groupby('period')['value'].agg(
                ['mean', 'median', 'std']
            ).reset_index()

            if not period_stats.empty:
                report["global_trends"][pollutant][index_name] = period_stats.to_dict('records')

    return report


def cluster_countries_by_pollutant(results: Dict, pollutant: str, n_clusters: int = 4) -> Dict:
    """
    Regroupe les pays selon leur profil pollution-richesse pour un polluant spécifique

    Args:
        results: Résultats de l'analyse précédente
        pollutant: Type de polluant à analyser
        n_clusters: Nombre de clusters à former

    Returns:
        Dictionnaire contenant les résultats du clustering
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import numpy as np

    # Convertir les données brutes en DataFrame
    df = pd.DataFrame(results["raw_data"])

    if df.empty:
        return {"error": "Pas de données disponibles pour le clustering"}

    # Filtrer pour le polluant spécifié
    df_pollutant = df[df['pollutant'] == pollutant]

    if df_pollutant.empty:
        return {"error": f"Pas de données disponibles pour le polluant {pollutant}"}

    # Identifier la dernière période disponible
    latest_period = sorted(df_pollutant['period'].unique())[-1]

    # Filtrer pour la dernière période
    df_latest = df_pollutant[df_pollutant['period'] == latest_period]

    if df_latest.empty:
        return {"error": f"Pas de données disponibles pour le polluant {pollutant} dans la période {latest_period}"}

    # Pivoter le DataFrame pour avoir les pays en lignes et les indices en colonnes
    df_pivot = df_latest.pivot_table(
        index='country',
        columns='index_name',
        values='value'
    )

    if df_pivot.empty:
        return {"error": f"Échec lors de la création de la table pivot pour le clustering"}

    # Sélectionner les indices pertinents
    indices_for_clustering = ['pollution_gdp_ratio', 'normalized_ratio', 'env_inequality_index']
    available_indices = [idx for idx in indices_for_clustering if idx in df_pivot.columns]

    if not available_indices:
        return {"error": "Aucun des indices requis n'est disponible pour le clustering"}

    # Extraire les données pour le clustering
    X = df_pivot[available_indices].values
    countries = df_pivot.index.tolist()

    # Imputer les valeurs manquantes
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Vérifier s'il reste des valeurs NaN ou inf
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print("Attention: Des valeurs NaN ou Inf persistent après imputation.")
        # Remplacer les valeurs restantes par 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Préparer les résultats du clustering
    clusters = {}
    for i in range(n_clusters):
        clusters[f"cluster_{i}"] = {
            "countries": [],
            "center": kmeans.cluster_centers_[i].tolist(),
            "profile": "",
            "count": 0
        }

    # Assigner les pays aux clusters
    for country, cluster_label in zip(countries, cluster_labels):
        clusters[f"cluster_{cluster_label}"]["countries"].append(country)
        clusters[f"cluster_{cluster_label}"]["count"] += 1

    # Déterminer le profil de chaque cluster
    feature_names = available_indices

    for cluster_id, cluster_info in clusters.items():
        center = cluster_info["center"]
        profiles = []

        for i, feature in enumerate(feature_names):
            if center[i] > 1.0:  # Valeur seuil arbitraire
                profiles.append(f"{feature} élevé")
            elif center[i] < -1.0:
                profiles.append(f"{feature} faible")

        if profiles:
            cluster_info["profile"] = ", ".join(profiles)
        else:
            cluster_info["profile"] = "profil moyen"

        # Calculer le pourcentage du total
        cluster_info["percentage"] = (cluster_info["count"] / len(countries)) * 100

    return {
        "pollutant": pollutant,
        "period": latest_period,
        "features_used": feature_names,
        "clusters": clusters
    }


def extract_indices_by_country(results: Dict) -> Dict:
    """
    Extrait les indices par pays à partir des résultats d'analyse

    Args:
        results: Dictionnaire des résultats d'analyse

    Returns:
        Dictionnaire contenant les indices par pays, structuré pour faciliter la visualisation
    """
    indices_by_country = {}

    for country_name, country_data in results["countries_data"].items():
        indices_by_country[country_name] = {
            "periods": {}
        }

        # Extraire les indices pour chaque période
        if "indices" in country_data:
            for period, period_data in country_data["indices"].items():
                indices_by_country[country_name]["periods"][period] = {}

                # Extraire les indices pour chaque polluant
                for pollutant, pollutant_indices in period_data.items():
                    indices_by_country[country_name]["periods"][period][pollutant] = pollutant_indices

    return indices_by_country


def extract_latest_indices_by_country(results: dict) -> dict:
    """
    Extrait les indices par pays pour la période la plus récente, en s'assurant
    que tous les pays sont inclus

    Args:
        results: Dictionnaire des résultats d'analyse

    Returns:
        Dictionnaire contenant les indices les plus récents par pays
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    latest_indices_by_country = {}

    # Récupérer toutes les périodes disponibles
    all_periods = set()
    for country_data in results["countries_data"].values():
        if "indices" in country_data:
            all_periods.update(country_data["indices"].keys())

    # Convertir en liste et trier pour obtenir la période la plus récente
    periods_list = sorted(list(all_periods))

    if not periods_list:
        print("Aucune période trouvée dans les données")
        return {}

    latest_period = periods_list[-1]
    print(f"Extraction des indices pour la période la plus récente: {latest_period}")

    # Pour chaque pays dans les données, essayer d'extraire ses indices pour la période la plus récente
    # Si non disponible, chercher la période la plus récente disponible pour ce pays
    for country_name, country_data in results["countries_data"].items():
        if "indices" not in country_data:
            continue

        # Vérifier si la période la plus récente est disponible pour ce pays
        if latest_period in country_data["indices"]:
            latest_indices_by_country[country_name] = {
                "period": latest_period,
                "pollutants": country_data["indices"][latest_period]
            }
        else:
            # Si la période la plus récente n'est pas disponible, prendre la dernière période disponible
            country_periods = sorted(list(country_data["indices"].keys()))
            if country_periods:
                country_latest = country_periods[-1]
                latest_indices_by_country[country_name] = {
                    "period": country_latest,
                    "pollutants": country_data["indices"][country_latest]
                }

    # Vérifier combien de pays ont été extraits
    print(f"Nombre de pays extraits: {len(latest_indices_by_country)}")

    return latest_indices_by_country


def extract_indices_by_country_all_periods(results: dict) -> dict:
    """
    Extrait les indices par pays pour toutes les périodes disponibles

    Args:
        results: Dictionnaire des résultats d'analyse

    Returns:
        Dictionnaire contenant les indices par pays et par période
    """
    indices_by_country = {}

    for country_name, country_data in results["countries_data"].items():
        indices_by_country[country_name] = {
            "periods": {}
        }

        # Extraire les indices pour chaque période
        if "indices" in country_data:
            for period, period_data in country_data["indices"].items():
                indices_by_country[country_name]["periods"][period] = period_data

    # Vérifier combien de pays ont été extraits
    print(f"Nombre de pays extraits (toutes périodes): {len(indices_by_country)}")

    return indices_by_country


def save_indices_files(results: dict, output_folder: str):
    """
    Sauvegarde plusieurs fichiers d'indices par pays, avec gestion des erreurs
    et conversion des types NumPy

    Args:
        results: Dictionnaire des résultats d'analyse
    """
    import json

    # Fonction pour convertir les types NumPy en types Python natifs
    def convert_numpy_types(obj):
        """Convertit récursivement les types NumPy en types Python natifs"""
        import numpy as np
        import pandas as pd

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(v) for v in obj)
        elif pd.isna(obj):
            return None
        return obj

    # 1. Tous les indices par pays et période
    all_indices = extract_indices_by_country_all_periods(results)
    try:
        with open(output_folder+'indices_by_country.json', 'w') as f:
            json.dump(convert_numpy_types(all_indices), f, indent=2)
        print("Indices par pays sauvegardés dans indices_by_country.json")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des indices par pays: {e}")

    # 2. Indices de la période la plus récente uniquement
    latest_indices = extract_latest_indices_by_country(results)
    try:
        with open(output_folder+'latest_indices_by_country.json', 'w') as f:
            json.dump(convert_numpy_types(latest_indices), f, indent=2)
        print("Indices les plus récents par pays sauvegardés dans latest_indices_by_country.json")

        # Liste des pays pour vérification
        country_list = sorted(list(latest_indices.keys()))
        print(f"Liste des premiers 10 pays (sur {len(country_list)}):")
        for i, country in enumerate(country_list[:10]):
            print(f"  {i + 1}. {country}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des indices les plus récents: {e}")

    # 3. Indices composite uniquement
    composite_indices = {}
    for country, data in all_indices.items():
        composite_indices[country] = {"periods": {}}

        for period, pollutants in data["periods"].items():
            if "composite" in pollutants:
                composite_indices[country]["periods"][period] = pollutants["composite"]

    try:
        with open(output_folder+'composite_indices_by_country.json', 'w') as f:
            json.dump(convert_numpy_types(composite_indices), f, indent=2)
        print("Indices composite par pays sauvegardés dans composite_indices_by_country.json")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des indices composite: {e}")


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys

    input_data_sensors = './data/air-quality/pollutant_data_completed.json'
    input_data_pib = './data/revenus-fiscaux/PIB/PIB_2016_2023.json'

    output_folder = 'resultats/world/'

    ignore_country = ["Turkey"]

    # Charger les données
    try:
        with open(input_data_sensors, 'r') as f:
            sensor_data = json.load(f)
            sensor_data = {country: data for country, data in sensor_data.items() if country not in ignore_country}
    except FileNotFoundError:
        print("Fichier sensor_data.json non trouvé.")
        sys.exit(1)

    try:
        with open(input_data_pib, 'r') as f:
            gdp_data = json.load(f)
            gdp_data = [country for country in gdp_data if country["Country Name"] not in ignore_country]
    except FileNotFoundError:
        print("Fichier gdp_data.json non trouvé.")
        sys.exit(1)

    # Analyser les données
    results = analyze_global_pollution_wealth_relation(sensor_data, gdp_data)

    # Générer un rapport
    report = generate_report(results)

    # Clustering des pays pour chaque polluant
    pollutant_types = identify_pollutant_types(sensor_data)
    clusters_by_pollutant = {}

    try:
        for pollutant in pollutant_types:
            clusters = cluster_countries_by_pollutant(results, pollutant)
            if "error" not in clusters:
                clusters_by_pollutant[pollutant] = clusters
    except ImportError:
        print("scikit-learn n'est pas installé. Le clustering n'a pas été effectué.")

    if clusters_by_pollutant:
        report["clusters_by_pollutant"] = clusters_by_pollutant

    # Sauvegarder les résultats
    save_results_to_json(results, output_folder+'multi_pollutant_analysis_results.json')
    save_results_to_json(report, output_folder+'multi_pollutant_analysis_report.json')

    save_indices_files(results, output_folder)

    print("Analyse terminée. Résultats sauvegardés dans multi_pollutant_analysis_results.json")
    print("Rapport sauvegardé dans multi_pollutant_analysis_report.json")
