import csv
import json
import os

def parse_csv(folder, file, output_file, target_column, extra_params={}):
    """Parses CSV file to include a metadata file.

    Args:
        folder (string): Folder relative
        file (string): File name (don't add .csv to name)
        output_file (string): File name (don't add .csv to name)
        target (string: target column
        extra_params (dict, optional): [description]. Defaults to {}.
    """
    params = {
        "numerical_threshold": 1000,
        **extra_params
    }
    unique = {}
    actual_row = []
    with open(f"{folder}/{file}.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                actual_row = row
                line_count += 1
            else:
                for i, r in enumerate(row):
                    if i not in unique:
                        unique[i] = set()
                    unique[i].add(r)
                line_count += 1
        print(f'Processed {line_count} lines.')
    fields = {}
    boolean_rows = []
    for i in unique:
        print(actual_row[i], len(unique[i]))
        if len(unique[i]) == 2:
            # YES NO
            fields[actual_row[i]] = {"type": "categorical"}
            boolean_rows.append(i)
        elif len(unique[i]) > params['numerical_threshold']:
            fields[actual_row[i]] = {"type": "numerical", "subtype": "float"}
        else:
            fields[actual_row[i]] = {"type": "categorical"}
            # print(actual_row[i])


    with open(f"{folder}/{file}.csv") as csv_file:
        with open(f"{folder}/{output_file}.csv", "w") as output:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_writer = csv.writer(output, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    csv_writer.writerow(row)
                    line_count += 1
                else:
                    csv_writer.writerow(row)
                    line_count += 1
            print(f'Processed {line_count} lines.')

    output = {
        "tables": {
            str(f"{output_file}"): {
                "fields": fields,
                "path": str(f"{output_file}.csv"),
                "target": target_column
            }
        }
    }

    os.system(f"echo '{json.dumps(output, indent=2)}' > {folder}/metadata.json")

if __name__ == "__main__":
    import sys
    folder, input_file, output_file, target = None, None, None, None
    if (len(sys.argv) > 1 and sys.argv[1] == "hazard"):
        folder = "datasets/Hazards"
        input_file = "train"
        output_file = "LibertyMutualHazard"
        target = "Hazard"
    else:
        folder = "datasets/" + input("Folder relative to dataset (i.e. Hazards): ")
        input_file = input("Input file -- without the .csv (i.e. train): ")
        output_file = input("Output file -- without the .csv (i.e. LibertyMutualHazard): ")
        target = input ("Target column (i.e. Hazard): ")

    folder = "datasets/Hazards" if not folder else folder
    input_file = "train" if not input_file else input_file
    output_file = "LibertyMutualHazard" if not output_file else output_file
    target = "Hazard" if not target else target
        
    parse_csv(folder, input_file, output_file, target)