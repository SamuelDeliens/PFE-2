import pandas as pd
import json
import os
import numpy as np


def safe_convert_to_numeric(value, default=0.0):
    """
    Safely convert value to numeric, handling various edge cases

    Args:
        value: Input value to convert
        default: Default value to return if conversion fails

    Returns:
        float or default value
    """
    if pd.isna(value):
        return default

    # Convert to string and strip whitespace
    str_value = str(value).strip()

    # Handle specific cases
    if str_value in ['', '.', 'n.c', 'n.d', 'na', 'NaN']:
        return default

    try:
        # Replace comma with dot for decimal separator
        str_value = str_value.replace(',', '.')
        return float(str_value)
    except (ValueError, TypeError):
        return default


def safe_convert_to_int(value, default=0):
    """
    Safely convert value to integer, handling various edge cases

    Args:
        value: Input value to convert
        default: Default value to return if conversion fails

    Returns:
        int or default value
    """
    # First convert to numeric
    numeric_value = safe_convert_to_numeric(value)

    try:
        return int(numeric_value)
    except (ValueError, TypeError):
        return default


def load_geographical_data(geo_file_path):
    """
    Load geographical data from CSV file

    Args:
        geo_file_path (str): Path to the geographical CSV file

    Returns:
        dict: Geographical information keyed by INSEE code
    """
    try:
        # Read the geographical CSV file
        df = pd.read_csv(geo_file_path,
                         delimiter=';',
                         encoding='utf-8',
                         dtype=str)

        # Convert to dictionary
        geo_dict = {}
        for _, row in df.iterrows():
            code_insee = str(row['Code INSEE']).strip()
            geo_dict[code_insee] = {
                'geo_point_2d': row['geo_point_2d'],
                'geo_shape': row['geo_shape']
            }

        return geo_dict
    except Exception as e:
        print(f"Error loading geographical data: {e}")
        return {}


def process_single_ircom_file(file_path: str):
    """
    Process a single IRCOM CSV file

    Args:
        file_path (str): Path to the input IRCOM CSV file

    Returns:
        dict: Processed data for the file
    """
    try:
        # Detect the delimiter (semicolon or comma)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            delimiter = ';' if ';' in first_line else ','

        # Read the CSV with custom parsing
        df = pd.read_csv(file_path,
                         delimiter=delimiter,
                         encoding='utf-8',
                         keep_default_na=False,  # Prevent pandas from converting to NaN
                         dtype=str)  # Read everything as string first

        # Ensure correct column names if needed
        original_columns = df.columns.tolist()
        expected_columns = [
            'INSEE_COM', 'Libellé de la commune', 'Nombre de foyers fiscaux',
            'Revenu fiscal de référence des foyers fiscaux', 'Impôt net (total)*',
            'Nombre de foyers fiscaux imposés',
            'Revenu fiscal de référence des foyers fiscaux imposés',
            'Traitements et salaires_Nombre de foyers concernés',
            'Traitements et salaires_Montant',
            'Retraites et pensions_Nombre de foyers concernés',
            'Retraites et pensions_Montant'
        ]

        # Adjust columns if they don't match expected length
        if len(original_columns) != len(expected_columns):
            print(f"Warning: Unexpected column count in {file_path}")
            print("Original columns:", original_columns)
            # Use original columns or fallback to expected
            columns_to_use = original_columns[:len(expected_columns)]
            df = df[columns_to_use]
            df.columns = expected_columns[:len(columns_to_use)]
        else:
            df.columns = expected_columns

        # Extract year from filename
        try:
            year = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
            year = int(year) if year.isdigit() else None
        except:
            year = None

        if year is None:
            print(f"Could not extract year from filename: {file_path}")
            return {}

        # Convert DataFrame to a dictionary with INSEE_COM as key
        result = {}
        for _, row in df.iterrows():
            # Skip rows with empty INSEE_COM
            if not row['INSEE_COM'] or pd.isna(row['INSEE_COM']):
                continue

            code_insee = str(row['INSEE_COM'])

            # Initialize result for this INSEE code if not exists
            if code_insee not in result:
                result[code_insee] = {
                    'nom_commune': str(row['Libellé de la commune']).strip(),
                    'annee': {}
                }

            # Add data for this year
            result[code_insee]['annee'][year] = {
                'foyers_fiscaux': {
                    'total': safe_convert_to_int(row['Nombre de foyers fiscaux']),
                    'imposes': safe_convert_to_int(row['Nombre de foyers fiscaux imposés'])
                },
                'revenus': {
                    'revenu_fiscal_reference_total': safe_convert_to_numeric(
                        row['Revenu fiscal de référence des foyers fiscaux']),
                    'revenu_fiscal_reference_imposes': safe_convert_to_numeric(
                        row['Revenu fiscal de référence des foyers fiscaux imposés']),
                    'impot_net_total': safe_convert_to_numeric(row['Impôt net (total)*'])
                },
                'sources_revenus': {
                    'traitements_salaires': {
                        'nombre_foyers': safe_convert_to_int(row['Traitements et salaires_Nombre de foyers concernés']),
                        'montant': safe_convert_to_numeric(row['Traitements et salaires_Montant'])
                    },
                    'retraites_pensions': {
                        'nombre_foyers': safe_convert_to_int(row['Retraites et pensions_Nombre de foyers concernés']),
                        'montant': safe_convert_to_numeric(row['Retraites et pensions_Montant'])
                    }
                }
            }

        return result

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}


def process_all_ircom_files(input_dir: str, output_dir: str, geo_file_path: str):
    """
    Process all IRCOM CSV files in a directory

    Args:
        input_dir (str): Directory containing IRCOM CSV files
        output_dir (str): Directory to save the consolidated JSON file
        geo_file_path (str): Path to the geographical reference file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load geographical data
    geo_data = load_geographical_data(geo_file_path)

    # Collect all CSV files
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.csv')
    ]

    # Consolidated result
    consolidated_result = {}

    # Process each file
    for file_path in sorted(csv_files):
        print(f"Processing file: {file_path}")
        file_data = process_single_ircom_file(file_path)

        # Merge data
        for code_insee, data in file_data.items():
            if code_insee not in consolidated_result:
                # Add geographical information if available
                if code_insee in geo_data:
                    data['geo'] = geo_data[code_insee]

                consolidated_result[code_insee] = data
            else:
                # Merge years if the INSEE code already exists
                consolidated_result[code_insee]['annee'].update(data['annee'])

    # Generate output file path
    output_file = os.path.join(output_dir, 'ircom_multi_year_processed.json')

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_result, f, ensure_ascii=False, indent=2)

    print(f"Consolidated IRCOM data. Saved to {output_file}")
    print(f"Total unique communes processed: {len(consolidated_result)}")

    return consolidated_result


def inspect_ircom_csv(file_path: str):
    """
    Inspect the structure of an IRCOM CSV file

    Args:
        file_path (str): Path to the CSV file
    """
    # Detect the delimiter
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        delimiter = ';' if ';' in first_line else ','

    # Read the CSV
    df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', dtype=str)

    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nTotal number of rows:", len(df))
    print("\nSample problematic values:")
    for col in df.columns:
        problematic_values = df[df[col].str.contains(r'[^\w\s\.\-,]', na=False)][col].unique()
        if len(problematic_values) > 0:
            print(f"\nProblematic values in {col}:")
            print(problematic_values)


# Example usage
if __name__ == '__main__':
    INPUT_DIR = './data/revenus-fiscaux/IRCOM/clean'  # Directory containing IRCOM CSV files
    OUTPUT_DIR = './data/revenus-fiscaux/IRCOM/json'  # Output directory for JSON
    GEO_FILE_PATH = './data/correspondance-code-insee-code-postal.csv'  # Path to geographical reference file

    # Optional: Inspect a specific CSV file's structure
    # inspect_ircom_csv('./path/to/specific/file.csv')

    # Process all IRCOM files in the directory
    process_all_ircom_files(INPUT_DIR, OUTPUT_DIR, GEO_FILE_PATH)