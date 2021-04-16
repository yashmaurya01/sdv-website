import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(folder, file):
    """Splits CSV file into 4 training and testing sets

    Args:
        folder (string): Relative folder
        file (string): CSV file (without .csv)
    """
    data = pd.read_csv(f'{folder}/{file}.csv', index_col=0)

    for i in range(1,5):
        train, test = train_test_split(data, test_size=0.2)
        train.to_csv(f'{folder}/{file}_train{i}.csv')
        test.to_csv(f'{folder}/{file}_test{i}.csv')


if __name__ == "__main__":
    import sys
    folder, output_file = None, None
    if (len(sys.argv) > 1 and sys.argv[1] == "hazard"):
        folder = "datasets/Hazards"
        output_file = "LibertyMutualHazard"
    else:
        folder = "datasets/" + input("Folder relative to dataset (i.e. Hazards): ")
        output_file = input("Output file -- without the .csv (i.e. LibertyMutualHazard): ")

    folder = "datasets/Hazards" if not folder else folder
    output_file = "LibertyMutualHazard" if not output_file else output_file
        
    split_csv(folder, output_file)