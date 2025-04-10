import pandas as pd
import os
import re
import glob


def clean_ircom_excel_to_csv(input_file, output_dir):
    match = re.search(r'revenus_(\d{4})', input_file)
    if match:
        year = match.group(1)
    else:
        year = "unknown"

    output_file = os.path.join(output_dir, f"IRCOM_{year}.csv")

    try:
        df = pd.read_excel(input_file, header=None)

        start_col = 0
        dep_row = None
        for i, row in df.iterrows():
            if dep_row is not None:
                break
            for j, cell in enumerate(row):
                if isinstance(cell, str) and cell.startswith("Dép."):
                    start_col = j
                    dep_row = i
                    break

        if dep_row is None:
            print(f"Erreur: Impossible de trouver la ligne avec 'Dép.' dans {input_file}")
            return

        headers1 = df.iloc[dep_row].tolist()
        headers2 = df.iloc[dep_row + 1].tolist()

        combined_headers = []
        for i in range(len(headers1)):
            if pd.isna(headers1[i]) and not pd.isna(headers2[i]):
                combined_headers.append(str(headers2[i]))
            elif not pd.isna(headers1[i]) and pd.isna(headers2[i]):
                combined_headers.append(str(headers1[i]))
            elif not pd.isna(headers1[i]) and not pd.isna(headers2[i]):
                combined_headers.append(f"{headers1[i]}_{headers2[i]}")
            else:
                combined_headers.append(f"col_{i}")

        data = df.iloc[(dep_row + 2):].copy()
        data.columns = combined_headers

        for col in range(start_col):
            data = data.drop(columns=[data.columns[col]])

        data = data.replace('n.c.', '')
        data = data.replace('.', '')

        if "Dép." in data.columns:
            data["Dép."] = data["Dép."].astype(str).str.strip()
        if "Commune" in data.columns:
            data["Commune"] = data["Commune"].astype(str).str.strip()

        data.to_csv(output_file, index=False)
        print(f"Conversion réussie ! Fichier {os.path.basename(input_file)} sauvegardé sous {output_file}")

    except Exception as e:
        print(f"Erreur lors du traitement de {input_file}: {str(e)}")


def process_ircom_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    patterns = [
        os.path.join(input_dir, "**", "ircom_communes_complet_revenus_*.xlsx"),
        os.path.join(input_dir, "**", "*ircom*", "*.xlsx")
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    files = list(set(files))

    if not files:
        print(f"Aucun fichier IRCOM trouvé dans {input_dir}")
        return

    print(f"Nombre de fichiers IRCOM trouvés: {len(files)}")

    for file in files:
        if not output_dir:
            output_dir = os.path.dirname(file)
        clean_ircom_excel_to_csv(file, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convertir les fichiers IRCOM Excel en CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Fichier ou répertoire d\'entrée contenant les fichiers des données IRCOM.')
    parser.add_argument('-o', '--output_dir', type=str, required=False, help='Répertoire de sortie pour les fichiers CSV.')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Erreur: Le fichier ou le répertoire d'entrée '{args.input}' n'existe pas.")
        exit(1)
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        print(f"Erreur: Le répertoire de sortie '{args.output_dir}' n'existe pas.")
        exit(1)

    if os.path.isfile(args.input):
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.input)
        clean_ircom_excel_to_csv(args.input, args.output_dir)

    elif os.path.isdir(args.input):
        process_ircom_files(args.input, args.output_dir)

    else:
        print(f"Erreur: Le chemin '{args.input}' n'est ni un fichier ni un répertoire.")
        exit(1)
