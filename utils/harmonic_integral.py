'''
Author: Okrio
Date: 2022-03-01 23:29:51
LastEditTime: 2022-03-02 00:22:48
LastEditors: Please set LastEditors
Description: refer to "HGCN-plus"
FilePath: /unofficial_HGCNplus/harmonic_integral.py
'''
# import sys
import torch
# import os
import librosa as lib

import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from drawer import plot_mesh


def make_integral_matrix_cos():
    factor = np.zeros((3600, 257))
    harmonic_loc = np.zeros((3600, 257))
    last_loc = 0
    for f in range(600, 4200):
        peak_last = 1.
        j = f - 600
        for k in range(1, int(80000 // f) + 1):
            loc = int(f * k / 80000 * 256)
            peak = 1 / np.sqrt(k)
            factor[j, loc] = peak
            harmonic_loc[j, loc] = 1.0

            if loc - last_loc > 1:
                num_iter = loc - last_loc + 1
                F_cos = np.cos(np.linspace(0, 2 * np.pi, num_iter))
                F = np.linspace(peak_last, peak, num_iter)
                for i in range(0, num_iter):
                    factor[j, i + last_loc] = F_cos[i] * F[i]
            else:
                factor[j, loc] += -(peak + peak_last) / 2
                factor[j, last_loc] += -(peak + peak_last) / 2

            last_loc = loc
            peak_last = peak
    plot_mesh(factor, 'harmonic_integrate_matrix_cos')
    plot_mesh(harmonic_loc, 'harmonic_loc_cos')
    plt.show()
    np.save("./harmonic_integrate_matrix_cos", factor)
    np.save("./harmonic_loc_cos", harmonic_loc)


def make_integral_matrix():
    factor = np.zeros((4200, 257))
    harmonic_loc = np.zeros((4200, 257))
    for f in range(600, 4200):
        last_loc = 0
        for k in range(1, int(80000 // f) + 1):
            compress_freq_loc = int(f * k / 80000 * 256)
            value = 1 / np.sqrt(k)
            factor[f, compress_freq_loc] += value
            harmonic_loc[f, compress_freq_loc] = 1.0
            # 谷结构建模
            if compress_freq_loc - last_loc > 1:
                if (last_loc + compress_freq_loc) % 2 != 0:
                    first_loc = int((last_loc + compress_freq_loc) // 2)
                    second_loc = first_loc + 1
                    factor[f, first_loc] += -0.5 * value
                    factor[f, second_loc] += -0.5 * value
                else:
                    loc = int((last_loc + compress_freq_loc) // 2)
                    factor[f, loc] += -1 * value
            # elif compress_freq_loc - last_loc == 1:
            else:
                factor[f, compress_freq_loc] = factor[
                    f, compress_freq_loc] - value * 0.5
                factor[f, last_loc] = factor[f, last_loc] - value * 0.5
            last_loc = compress_freq_loc
    # plot_mesh(factor)
    # plot_mesh(harmonic_loc)
    np.save("harmonic_integrate_matrix", factor)
    np.save("harmonic_loc", harmonic_loc)
    return factor


def load_test(path, loc_path):
    smooth = nn.AvgPool1d(kernel_size=1, stride=1, padding=0)
    corr_factor = torch.tensor(np.load(path), dtype=torch.float).unsqueeze(0)
    loc = torch.tensor(np.load(loc_path), dtype=torch.float).unsqueeze(0)
    corr = corr_factor
    corr[corr != corr] = 0
    # plot_mesh(corr[0])
    noisy_path = r"wavs/fileid10_cleanBAC009S0657W0284_noiseuI44_PzWnCA_snr5_level-19.wav"
    clean_path = r"wavs/fileid10_BAC009S0657W0284.wav"
    noisy, _ = lib.load(noisy_path, sr=16000)
    clean, _ = lib.load(clean_path, sr=16000)
    noisy_stft = lib.stft(noisy, win_length=512, n_fft=512, hop_length=128)
    noisy_mag, noisy_phase = lib.magphase(noisy_stft)
    clean_stft = lib.stft(clean, win_length=512, hop_length=128, n_fft=512)
    clean_mag, clean_phase = lib.magphase(clean_stft)
    noisy_mag = noisy_mag**0.5
    clean_mag = clean_mag**0.5
    # noisy_stft = torch.stft(torch.Tensor(noisy),
    #                         win_length=512,
    #                         n_fft=512,
    #                         hop_length=128,
    #                         window=torch.hann_window(512))
    # clean_stft = torch.stft(torch.Tensor(clean),
    #                         win_length=512,
    #                         hop_length=128,
    #                         n_fft=512,
    #                         window=torch.hann_window(512))
    # noisy_mag = torch.sqrt(noisy_stft[:, :, 0]**2 + noisy_stft[:, :, 1]**2 +
    #                        1e-8)
    # clean_mag = torch.sqrt(clean_stft[:, :, 0]**2 + clean_stft[:, :, 1]**2 +
    #                        1e-8)
    # noisy_stft = stft_512(torch.tensor([noisy]))
    # noisy_mag = mag(noisy_stft)
    # clean_stft = stft_512(torch.tensor([clean]))
    # clean_mag = mag(clean_stft)
    plot_mesh(noisy_mag, "noisy_mag")
    plot_mesh(clean_mag, "clean_mag")
    harmonic_noisy = torch.matmul(corr, torch.tensor(noisy_mag))
    # harmonic_clean = torch.matmul(corr, clean_mag)
    plot_mesh(harmonic_noisy[0].data, "harmonic_nominee_noisy")
    # plot_mesh(harmonic_clean[0].data, "harmonic_nominee_clean")
    value, position = torch.topk(harmonic_noisy, k=5, dim=1)
    choosed_harmonic = torch.zeros(1, 257, noisy_stft.shape[-1])
    for i in range(1):
        choose = smooth(position.to(
            torch.float)[:, i, :].unsqueeze(1)).flatten().to(torch.long)
        # choosed_harmonic += loc[:, choose, :].permute(0, 2, 1) # for HGCN
        choosed_harmonic += corr_factor[:,
                                        choose, :].permute(0, 2,
                                                           1)  # for HGCN-plus
    # choosed_result = (choosed_harmonic > 0).to(torch.float) # for HGCN
    choosed_result = choosed_harmonic.to(torch.float)  # for HGCN-plus
    plot_mesh(choosed_result[0].data, "harmonic position predicted by noisy")
    plt.show()


if __name__ == "__main__":
    # make_integral_matrix_cos()
    load_test('./harmonic_integrate_matrix_cos.npy', './harmonic_loc_cos.npy')
    print('sc')
