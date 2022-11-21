# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import argparse
from typing import Union, List
import numpy as np
import json
from pathlib import Path
import torchaudio
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-pd", "--project_dir", required=True, help="path to current project root")
parser.add_argument("-tr", "--train_dir", required=True, help="path to dir containing all train audio files")
parser.add_argument("-tt", "--test_dir", required=True, help="path to dir containing all test audio files")
parser.add_argument("-vp", "--val_p", default=["p286", "p287"], type=Union[float, List[str]], required=False,
                    help="percentage of train files to be used for validation, this could also accept a list of str, "
                         "corresponding to prefixes to be assigned to validation set")
parser.add_argument("-bfvs", "--brute_force_validation_split", required=False, default=False,
                    help="if True, assuming val_p is float, divides train and val set by number of audio segment")
parser.add_argument("-n", "--dataset_name", required=True, help="specifies egs/<dataset_name>/{tr/val/tt}.json "
                                                                "path to exported files")
args = parser.parse_args()


def create_lists_for_jsons():
    train, val, test = [], [], []

    # loop over train_dir
    train_dir = args.train_dir
    if isinstance(args.val_p, float) and args.brute_force_validation_split:
        files = list(os.listdir(train_dir))
        np.random.shuffle(files)
        n = int(len(files) * args.val_p)
        for idx, file_set in enumerate([files[:-n], files[-n:]]):
            for file in files[:-n]:
                filepath = f"{train_dir}/{file}"
                info = torchaudio.info(filepath)
                if hasattr(info, 'num_frames'):
                    # new version of torchaudio
                    tmp = [filepath, info.num_frames]
                else:
                    siginfo = info[0]
                    tmp = [filepath, siginfo.length // siginfo.channels]
                if idx == 0:
                    train.append(tmp)
                else:
                    val.append(tmp)
    else:
        if isinstance(args.val_p, float):
            speakers = list(set([f.split("_")[0] for f in os.listdir(train_dir)]))
            np.random.shuffle(speakers)
            num_speakers_in_val = int(args.val_p * len(speakers))
            val_files = speakers[:num_speakers_in_val]
        else:
            val_files = args.val_p
        for file in tqdm(os.listdir(train_dir), desc="Train set"):
            filepath = f"{train_dir}/{file}"
            info = torchaudio.info(filepath)
            if hasattr(info, 'num_frames'):
                # new version of torchaudio
                tmp = [filepath, info.num_frames]
            else:
                siginfo = info[0]
                tmp = [filepath, siginfo.length // siginfo.channels]
            if True in [file.startswith(pref) for pref in val_files]:
                val.append(tmp)
            else:
                train.append(tmp)

    # loop over test dir
    test_dir = args.test_dir
    for file in tqdm(os.listdir(test_dir), desc="Test set"):
        filepath = f"{test_dir}/{file}"
        info = torchaudio.info(filepath)
        if hasattr(info, 'num_frames'):
            # new version of torchaudio
            test.append([filepath, info.num_frames])
        else:
            siginfo = info[0]
            test.append([filepath, siginfo.length // siginfo.channels])

    return train, val, test


def create_dirs_and_save_jsons(train, val, test):
    directory = Path(args.project_dir).joinpath("egs").joinpath(args.dataset_name)
    os.makedirs(directory, exist_ok=True)
    for ss, ss_name in zip([train, val, test], ["tr", "cv", "tt"]):
        with open(directory.joinpath(f"{ss_name}.json").__str__(), "w") as f:
            f.write(json.dumps(ss, indent=4))


if __name__ == "__main__":
    tr, cv, tt = create_lists_for_jsons()
    create_dirs_and_save_jsons(tr, cv, tt)