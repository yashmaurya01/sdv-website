import os
from generate import sample

def preprocess_hazards(x):
    x = x.drop("Id", 1)
    return x

if __name__ == "__main__":
    for i in range(1, 5):
        os.system(f"rm datasets/Hazards/LibertyMutualHazard.csv")
        os.system(f"cp datasets/Hazards/LibertyMutualHazard_train{i}.csv datasets/Hazards/LibertyMutualHazard.csv")
        sample.sample_tablegan("Hazards", "LibertyMutualHazard", "./datasets", output=f"datasets/Hazards/LibertyMutualHazard_train_output{i}.csv", sample_synthetic_rows=41600, preprocess_table=preprocess_hazards)
        sample.sample_ctgan("Hazards", "LibertyMutualHazard", "./datasets", output=f"datasets/Hazards/LibertyMutualHazard_train_output{i}.csv", sample_synthetic_rows=41600)
