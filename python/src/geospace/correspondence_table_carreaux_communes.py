import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# 1. Charger les données géographiques et les carreaux
year = [2015, 2017, 2019]
carreaux_csv = pd.read_csv("./data/revenus-fiscaux/Filosofi/harmonized/harmonized_Filosofi2017_carreaux_1km_met.csv", sep=',')
communes = gpd.read_file("./data/commune-frmetdrom/COMMUNE_FRMETDROM.shp")
id_col = 'Idcar_1km'

# Vérifiez les CRS
print(f"CRS des communes: {communes.crs}")


# 2. Créez les géométries des carreaux
def create_grid_cell_from_id(id_inspire):
    # Pour les identifiants du format CRS3035RES200mN2029800E4252400
    try:
        parts = id_inspire.split('E')
        x = int(parts[-1])
        y = int(parts[-2].split('N')[-1])

        # Déterminer la taille du carreau (200m ou 1km)
        size = 200 if 'RES200m' in id_inspire else 1000

        return box(x, y, x + size, y + size)
    except:
        print(f"Problème avec l'ID: {id_inspire}")
        return None


# Créer un GeoDataFrame pour les carreaux
carreaux = carreaux_csv.copy()
carreaux['geometry'] = carreaux[id_col].apply(create_grid_cell_from_id)
carreaux = gpd.GeoDataFrame(carreaux, geometry='geometry', crs="EPSG:3035")  # CRS correspondant au format des IDs

# Vérification : imprimez quelques géométries pour vous assurer qu'elles sont créées correctement
print("Premières géométries des carreaux:")
print(carreaux['geometry'].head())

# 3. Calculer la surface de chaque carreau AVANT l'intersection
carreaux['area_total'] = carreaux.geometry.area

# 3. Assurez-vous que les deux couches sont dans le même CRS
communes = communes.to_crs(carreaux.crs)

# 4. Vérifiez l'étendue des deux couches pour détecter des problèmes d'échelle ou de position
print(f"Étendue des carreaux: {carreaux.total_bounds}")
print(f"Étendue des communes: {communes.total_bounds}")

# 5. Effectuez l'intersection
# Utilisez sjoin au lieu de overlay pour un débogage plus facile
intersection = gpd.sjoin(carreaux, communes, how="inner", predicate="intersects")

# Vérifiez si l'intersection contient des données
print(f"Nombre de lignes dans l'intersection: {len(intersection)}")


# Si l'intersection est vide, essayez avec un buffer
if len(intersection) == 0:
    print("Essai avec un petit buffer...")
    carreaux['geometry'] = carreaux['geometry'].buffer(1)  # Buffer de 1 mètre
    intersection = gpd.sjoin(carreaux, communes, how="inner", predicate="intersects")
    print(f"Après buffer, nombre de lignes: {len(intersection)}")

# 6. Calculez les poids si l'intersection contient des données
if len(intersection) > 0:

    # Joindre les géométries d'origine pour calculer l'intersection précise
    intersection = gpd.overlay(carreaux, communes, how='intersection')
    intersection['area_intersection'] = intersection.geometry.area

    area_info = carreaux[[id_col, 'area_total']].copy()

    # Fusionner avec les surfaces totales
    intersection = pd.merge(intersection, area_info, on=id_col, how='left',
                            suffixes=('_inter', '_total'))
    print("Intersection head: ", intersection.head())

    # Calculer les poids
    intersection['weight'] = intersection['area_intersection'] / intersection['area_total_total']

    # Créer la table finale en utilisant les colonnes disponibles
    # Vérifiez d'abord quelles colonnes sont présentes
    print("Colonnes disponibles dans l'intersection:")
    print(intersection.columns.tolist())

    # Utilisez les colonnes appropriées pour le code commune (peut être INSEE_COM au lieu de Depcom)
    if 'INSEE_COM' in intersection.columns:
        commune_col = 'INSEE_COM'
    elif 'code_insee' in intersection.columns:
        commune_col = 'code_insee'
    else:
        # Afficher toutes les colonnes pour identifier celle qui contient les codes communes
        print("Colonnes disponibles:")
        print(intersection.columns.tolist())
        # Choisir une colonne qui semble contenir les codes communes
        commune_col = input("Entrez le nom de la colonne contenant les codes communes: ")

    correspondence_table = intersection[[id_col, commune_col, 'weight']]

    correspondence_table.to_csv('./data/correspondence_table_carreaux_communes_2017.csv', index=False)
    print(f"Table de correspondance créée avec succès: {len(correspondence_table)} lignes.")
else:
    print("Aucune intersection trouvée entre les carreaux et les communes!")