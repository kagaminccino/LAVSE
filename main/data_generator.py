import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import random

random.seed('LAVSE')

def data_nor(data, channel):
    mean = torch.mean(data, channel, keepdim=True)
    std = torch.std(data, channel, keepdim=True) + 1e-12
    nor_data = (data - mean) / std

    return nor_data, mean, std

def stft2spec(stft, normalized, save_phase, save_mean_std):
    magnitude = torch.norm(stft, 2, -1)

    if save_phase:
        # (1, 257, frames, 2) -> (257, frames, 2) -> (2, 257, frames)
        stft = stft.squeeze(0)
        stft = stft.permute(2, 0, 1)

        phase = stft / (magnitude + 1e-12)

        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), phase
        else:
            return (specgram, None, None), phase

    else:
        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), None
        else:
            return (specgram, None, None), None

class AV_Dataset(Dataset):
    def __init__(self, name, data_path_list=None, mode='no_model', av=False):
        self.name = name # name: 'train', 'val', 'test', 'clean'
        self.mode = mode # mode: 'training', 'validation', 'testing', 'no_model'
        self.av = av
        self.samples = []

        for dir_path in data_path_list:
            clean_path, noisy_path, lip_path = dir_path
            stftpt_paths = sorted(glob.glob(noisy_path + '*.pt'))

            for stftpt_path in stftpt_paths:
                file_name = stftpt_path.rsplit('.', 1)[0]
                file_name = file_name.rsplit('/', 2)[-1]

                clean_stftpt_path = stftpt_path.replace(noisy_path, clean_path)
                noisy_stftpt_path = stftpt_path
                lippt_path = lip_path + file_name + '.pt'

                self.samples.append((clean_stftpt_path, noisy_stftpt_path, lippt_path))

        if self.mode == 'training':
            random.shuffle(self.samples)
            self.samples = self.samples[:12000]
        elif self.mode == 'validation':
            random.shuffle(self.samples)
            self.samples = self.samples[:800]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_stftpt_path, noisy_stftpt_path, lippt_path = self.samples[idx]

        file_name = noisy_stftpt_path.rsplit('.', 1)[0]
        file_name = file_name.rsplit('/', 2)
        noisy_type = file_name[-2]
        file_name = file_name[-1]
        file_name = file_name + '__' + noisy_type

        if self.mode == 'training' or self.mode == 'validation':
            stft_clean = torch.load(clean_stftpt_path)
            stft_noisy = torch.load(noisy_stftpt_path)

            (spec_clean, _, _), _ = stft2spec(stft_clean, normalized=False, save_phase=False, save_mean_std=False)
            (spec_noisy, _, _), _ = stft2spec(stft_noisy, normalized=True, save_phase=False, save_mean_std=False)

            if self.av:
                lippt = torch.load(lippt_path)

                frame_num = min(spec_clean.shape[-1], lippt.shape[-1])

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                data = file_name, frame_num, spec_clean[..., :frame_num], spec_noisy[..., :frame_num], None, None, lippt[..., :frame_num]

            else:
                frame_num = spec_clean.shape[-1]

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                data = file_name, frame_num, spec_clean[..., :frame_num], spec_noisy[..., :frame_num], None, None, None

        elif self.mode == 'testing' or self.mode == 'no_model':
            if self.mode == 'testing':
                stft_noisy = torch.load(noisy_stftpt_path)
                (spec_noisy, _, _), phase = stft2spec(stft_noisy, normalized=True, save_phase=True, save_mean_std=False)

            elif self.mode == 'no_model':
                stft_noisy = torch.load(noisy_stftpt_path)
                (spec_noisy, spec_noisy_mean, spec_noisy_std), phase = stft2spec(stft_noisy, normalized=True, save_phase=True, save_mean_std=True)

            if self.av:
                lippt = torch.load(lippt_path)

                frame_num = min(spec_noisy.shape[-1], lippt.shape[-1])

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                if self.mode == 'no_model':
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], spec_noisy_mean, spec_noisy_std, lippt[..., :frame_num]
                else:
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], None, None, lippt[..., :frame_num]

            else:
                frame_num = spec_noisy.shape[-1]

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                if self.mode == 'no_model':
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], spec_noisy_mean, spec_noisy_std, None
                else:
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], None, None, None

        return data
