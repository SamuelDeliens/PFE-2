import pandas as pd
import json
import numpy as np
import os
from typing import Dict, Any


def process_pollution_data(pollution_files: list, geo_info_file: str) -> Dict[str, Any]:
    """
    Process pollution data from multiple files and generate a consolidated JSON

    Args:
        pollution_files (list): List of paths to pollution CSV files
        geo_info_file (str): Path to the geographical information CSV file

    Returns:
        Dict containing processed data
    """
    # Read geographical information
    try:
        geo_df = pd.read_csv(geo_info_file, sep=';', encoding='utf-8')
        geo_df.set_index('Code INSEE', inplace=True)
    except Exception as e:
        print(f"Error reading geographical info: {e}")
        geo_df = pd.DataFrame()

    # Consolidated result
    result = {}

    # Process each pollution file
    for pollution_file in pollution_files:
        try:
            # Read the CSV file
            pollution_df = pd.read_csv(pollution_file, parse_dates=[0])

            # Rename the first column to 'datetime'
            pollution_df = pollution_df.rename(columns={pollution_df.columns[0]: 'datetime'})

            # Melt the dataframe to long format
            melted_df = pollution_df.melt(id_vars=['datetime'], var_name='Code INSEE', value_name='Pollution')

            # Remove rows with NaN values
            melted_df = melted_df.dropna(subset=['Pollution'])

            # Group by month and INSEE code
            monthly_agg = melted_df.groupby([
                pd.Grouper(key='datetime', freq='M'),
                'Code INSEE'
            ])['Pollution'].agg([
                ('mean', 'mean'),
                ('min', 'min'),
                ('max', 'max'),
                ('median', 'median'),
                ('q1', lambda x: x.quantile(0.25)),
                ('q3', lambda x: x.quantile(0.75)),
                ('count', 'count')
            ]).reset_index()

            # Update the result dictionary
            for _, row in monthly_agg.iterrows():
                code_insee = row['Code INSEE']
                month = row['datetime'].strftime('%Y-%m')

                # Get geographical info if available
                geo_info = geo_df.loc[code_insee] if not geo_df.empty and code_insee in geo_df.index else {}

                if code_insee not in result:
                    result[code_insee] = {
                        'nom': geo_info.get('Commune', ''),
                        'geo_point_2d': geo_info.get('geo_point_2d', ''),
                        'geo_shape': geo_info.get('geo_shape', ''),
                        'data': {}
                    }

                result[code_insee]['data'][month] = {
                    'mean': round(row['mean'], 2) if pd.notnull(row['mean']) else None,
                    'min': round(row['min'], 2) if pd.notnull(row['min']) else None,
                    'max': round(row['max'], 2) if pd.notnull(row['max']) else None,
                    'median': round(row['median'], 2) if pd.notnull(row['median']) else None,
                    'q1': round(row['q1'], 2) if pd.notnull(row['q1']) else None,
                    'q3': round(row['q3'], 2) if pd.notnull(row['q3']) else None,
                    'count': row['count']
                }

        except Exception as e:
            print(f"Error processing {pollution_file}: {e}")

    return result


def process_all_years(pollution_dir: str, geo_info_file: str, output_dir: str):
    """
    Process pollution data for all years in a directory

    Args:
        pollution_dir (str): Directory containing pollution CSV files
        geo_info_file (str): Path to the geographical information CSV file
        output_dir (str): Directory to save output JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Collect all CSV files
    pollution_files = [
        os.path.join(pollution_dir, filename)
        for filename in os.listdir(pollution_dir)
        if filename.endswith('.csv')
    ]

    # Sort files to ensure consistent processing
    pollution_files.sort()

    # Process all collected files
    processed_data = process_pollution_data(pollution_files, geo_info_file)

    # Save to a single JSON file
    output_file = os.path.join(output_dir, 'pollution_processed_all_years.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processed all files and saved to {output_file}")
    print(f"Total number of INSEE codes processed: {len(processed_data)}")


# Debugging function to inspect CSV structure
def inspect_csv(file_path: str):
    """
    Inspect the structure of a CSV file
    """
    df = pd.read_csv(file_path)
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)


if __name__ == '__main__':
    # Set your directory paths
    POLLUTION_DIR = './data/air-quality/idf/final/PM25/'
    GEO_INFO_FILE = './data/correspondance-code-insee-code-postal.csv'
    OUTPUT_DIR = './data/air-quality/idf/json/'

    # Process all years
    process_all_years(POLLUTION_DIR, GEO_INFO_FILE, OUTPUT_DIR)