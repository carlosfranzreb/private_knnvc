"""
After anonymizing all the data, create the datafiles for the VPC2024 project.
"""

import os
import json
from argparse import ArgumentParser


SPKANON_DIR = "/path/to/spkanon_eval"
VPC_DIR = "/path/to/Voice-Privacy-Challenge-2024/data"


def create_datafiles(suffix: str, data_dir: str):
    # Copy existing directories and files
    os.system(f"bash {SPKANON_DIR}/vpc2024/create_dirs.sh {suffix} mcadams")

    # open the datafile with the anonymized data
    df = os.path.join(data_dir, "anon_eval.txt")
    anon_data = dict()
    for line in open(df):
        path = json.loads(line)["path"]
        filename = os.path.splitext(os.path.basename(path))[0]
        anon_data[filename] = path
    print(f"Anonymized data: {len(anon_data)}")

    for folder in os.listdir(VPC_DIR):
        if not folder.endswith(suffix):
            continue
        print(f"Processing {folder}")
        wav_file = os.path.join(VPC_DIR, folder, "wav.scp")
        backup = os.path.join(VPC_DIR, folder, "wav.scp.bak")
        os.system(f"mv {wav_file} {backup}")
        reader = open(backup)
        writer = open(wav_file, "w")
        for line in reader:
            fname, path = line.strip().split()
            new_path = os.path.join(SPKANON_DIR, anon_data[fname])
            writer.write(f"{fname} {new_path}\n")
        reader.close()
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("suffix", type=str, help="Suffix of the data directory")
    parser.add_argument("data_dir", type=str, help="Name of the data directory")
    args = parser.parse_args()

    create_datafiles(args.suffix, args.data_dir)
