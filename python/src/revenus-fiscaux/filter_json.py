import json
import os


def extract_insee_codes_from_json_files(json_directory):
    """
    Extract unique INSEE codes from all JSON files in a directory

    Args:
        json_directory (str): Path to directory containing JSON files

    Returns:
        set: Unique INSEE codes found across all JSON files
    """
    # Set to store unique INSEE codes
    all_insee_codes = set()

    # List all JSON files in the directory
    json_files = [
        os.path.join(json_directory, f)
        for f in os.listdir(json_directory)
        if f.endswith('.json')
    ]

    print(f"Found {len(json_files)} JSON files to process")

    # Process each JSON file
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Add keys (assuming INSEE codes) to the set
                file_codes = set(data.keys())
                all_insee_codes.update(file_codes)

                print(f"Processed {file_path}: {len(file_codes)} codes found")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Total unique INSEE codes found: {len(all_insee_codes)}")
    return all_insee_codes


def filter_ircom_data(json_directory, ircom_json_path, output_path):
    """
    Filter IRCOM data to include only INSEE codes present in JSON files

    Args:
        json_directory (str): Path to directory containing reference JSON files
        ircom_json_path (str): Path to the full IRCOM JSON file
        output_path (str): Path to save the filtered IRCOM JSON
    """
    # Extract INSEE codes from all JSON files in the directory
    selected_codes = extract_insee_codes_from_json_files(json_directory)

    # Load full IRCOM data
    try:
        with open(ircom_json_path, 'r', encoding='utf-8') as f:
            ircom_data = json.load(f)

    except Exception as e:
        print(f"Error reading IRCOM JSON: {e}")
        return

    # Filter IRCOM data
    filtered_ircom_data = {}
    not_selected_codes = selected_codes
    for code, data in ircom_data.items():
        if code in selected_codes:
            filtered_ircom_data[code] = data
            not_selected_codes.remove(code)

    print(f"Not Filtered IRCOM data: {not_selected_codes} codes found")

    print(f"Number of INSEE codes in filtered IRCOM data: {len(filtered_ircom_data)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save filtered data
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_ircom_data, f, ensure_ascii=False, indent=2)

        print(f"Filtered IRCOM data saved to {output_path}")

    except Exception as e:
        print(f"Error saving filtered IRCOM data: {e}")


# Example usage
if __name__ == '__main__':
    # Set your input and output paths
    JSON_DIRECTORY = './data/air-quality/France/'
    IRCOM_JSON_PATH = './data/revenus-fiscaux/IRCOM/json/ircom_multi_year_processed.json'
    OUTPUT_PATH = './data/revenus-fiscaux/IRCOM/json/ircom_filtered_France.json'

    filter_ircom_data(JSON_DIRECTORY, IRCOM_JSON_PATH, OUTPUT_PATH)