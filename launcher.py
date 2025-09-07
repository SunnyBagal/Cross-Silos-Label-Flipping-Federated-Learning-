#launcher.py:


import subprocess
import pandas as pd
import shlex

# Get valid hospital names from the dataset directly
def get_valid_hospitals(csv_path="data/shortened_healthcare_dataset_random_hospitals.csv"):
    df = pd.read_csv(csv_path)
    return df["Hospital"].dropna().astype(str).str.strip().unique().tolist()

if __name__ == "__main__":
    hospitals = get_valid_hospitals()
    print("Launching clients for hospitals:", hospitals)

    procs = []
    for hospital in hospitals:
        # Use bash -c so we can export environment variable then run python in same subprocess
        cmd = f'export HOSPITAL_NAME="{hospital}" && python3 client.py'
        p = subprocess.Popen(["/bin/bash", "-c", cmd])
        procs.append(p)

    # Optionally keep script alive until subprocesses exit (comment out to fire-and-forget)
    for p in procs:
        p.wait()
