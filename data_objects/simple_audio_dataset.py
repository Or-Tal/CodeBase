# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
from typing import Union
import json
import torchaudio
import torch.nn.functional as F
import math


class SimpleAudioDataset:

    def construct_json(self, key_for_json_file):
        with open(getattr(self.cfg, key_for_json_file), "r") as f:
            files = json.load(f)
        return files

    def count_num_examples(self):
        cur_idx = 0
        idx_map = dict()
        for file, file_length in self.files:
            file_length = int(file_length)
            print(f"fl: {type(file_length)} | loass: {type(self.length_of_a_single_sample)}")
            if self.ignore_length:  # case where we iterate over complete samples
                idx_map[cur_idx] = (file, 0)
                cur_idx += 1
            elif file_length < self.length_of_a_single_sample and self.cfg.pad:  # short sample and pad is true
                idx_map[cur_idx] = (file, 0)
                cur_idx += 1
            elif self.cfg.pad:
                n = cur_idx + int(math.ceil((file_length - self.length_of_a_single_sample) / self.stride) + 1)
                i = 0
                while cur_idx < n:
                    idx_map[cur_idx] = (file, i)
                    cur_idx += 1
                    i += 1
            else:
                n = (file_length - self.length_of_a_single_sample) // self.stride + 1
                i = 0
                while cur_idx < n:
                    idx_map[cur_idx] = (file, i)
                    cur_idx += 1
                    i += 1
        return cur_idx, idx_map

    def __init__(self, dataset_config, key_for_json_file, ignore_length=False, include_path=False):
        self.cfg = dataset_config
        self.files = self.construct_json(key_for_json_file)  # this dummy assumes a list of (path, length) tuples
        self.ignore_length = ignore_length
        self.include_path = include_path
        self.length_of_a_single_sample = dataset_config.override_segment_length or \
                                         int(self.cfg.sample_rate * self.cfg.segment)
        self.stride = dataset_config.override_stride_length or int(self.cfg.sample_rate * self.cfg.stride)
        # store map in mem: index -> (file_path, segment_index_in_sample, length of the signal)
        self.num_samples, self.idx_to_sample_map = self.count_num_examples()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        file, segment_idx = self.idx_to_sample_map[index]
        flag = torchaudio.get_audio_backend() in ['soundfile', 'sox_io']
        offset = 0 if self.ignore_length else self.stride * segment_idx
        length_of_a_single_sample = 0 if self.ignore_length else self.length_of_a_single_sample
        kwargs = {'frame_offset' if flag else 'offset': offset,
                  "num_frames": length_of_a_single_sample or -1 if flag else length_of_a_single_sample}
        out, sr = torchaudio.load(str(file), **kwargs)

        # validation check
        target_sr, target_channels = self.cfg.sample_rate or sr, self.cfg.channels or out.shape[0]
        assert target_sr == sr, f"Expected {file} to have sample rate of {target_sr}, but got {sr}"
        assert out.shape[
                   0] == target_channels, f"Expected {file} to have channels of {target_channels}, but got {out.shape[0]}"

        if length_of_a_single_sample and length_of_a_single_sample > out.shape[-1]:
            out = F.pad(out, (0, length_of_a_single_sample - out.shape[-1]))

        return (out, file) if self.include_path else out
