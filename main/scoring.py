# PESQ
import subprocess
# STOI
import soundfile as sf
from pystoi.stoi import stoi

import glob
from tqdm import tqdm
import csv

from utils import cal_time
import time

def scoring_PESQ(clean_wav_path, noisy_wav_path, enhan_wav_path, sr):
    # print('clean_wav_path =', clean_wav_path)
    # print('noisy_wav_path =', noisy_wav_path)
    # print('before replace enhan_wav_path =', enhan_wav_path)
    enhan_wav_path = enhan_wav_path.replace('(', '\(')
    enhan_wav_path = enhan_wav_path.replace(')', '\)')
    # print(' after replace enhan_wav_path =', enhan_wav_path)

    noisy_pesq = subprocess.check_output('./PESQ +%d %s %s' % (sr, clean_wav_path, noisy_wav_path), shell=True)
    noisy_pesq = noisy_pesq.decode("utf-8")
    noisy_pesq = noisy_pesq.splitlines()[-1]
    noisy_pesq = noisy_pesq[-5:]
    noisy_pesq = float(noisy_pesq)

    enhan_pesq = subprocess.check_output('./PESQ +%d %s %s' % (sr, clean_wav_path, enhan_wav_path), shell=True)
    enhan_pesq = enhan_pesq.decode("utf-8")
    enhan_pesq = enhan_pesq.splitlines()[-1]
    enhan_pesq = enhan_pesq[-5:]
    enhan_pesq = float(enhan_pesq)

    return noisy_pesq, enhan_pesq

def scoring_STOI(clean, noisy, enhan, sr):
    noisy_stoi = stoi(clean, noisy, sr, extended=False)
    enhan_stoi = stoi(clean, enhan, sr, extended=False)

    return noisy_stoi, enhan_stoi

def scoring_file(clean_wav_path, noisy_wav_path, enhan_wav_path):

    file_name = noisy_wav_path.rsplit('.', 1)[0]
    file_name = file_name.rsplit('/', 2)
    noise_type = file_name[-2]
    file_name = file_name[-1]

    # print('noise_type =', noise_type)
    # print('file_name =', file_name)

    file_name = file_name + '__' + noise_type
    # print('file_name =', file_name)

    clean, sr = sf.read(clean_wav_path)
    noisy, sr = sf.read(noisy_wav_path)
    enhan, sr = sf.read(enhan_wav_path)

    noisy_pesq, enhan_pesq = scoring_PESQ(clean_wav_path, noisy_wav_path, enhan_wav_path, sr)
    noisy_stoi, enhan_stoi = scoring_STOI(clean, noisy, enhan, sr)

    return file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi

def scoring_dir(scoring_path_list):
    scores = []

    for dir_path in scoring_path_list:
        clean_path, noisy_path, enhan_path = dir_path

        print('Scoring PESQ and STOI of wav in \'' + noisy_path + '\'\n' + \
              '                             and \'' + enhan_path + '\'...')
        for wav_path in tqdm(sorted(glob.glob(noisy_path + '*.wav'))):

            clean_wav_path = wav_path.replace(noisy_path, clean_path)
            noisy_wav_path = wav_path
            enhan_wav_path = wav_path.replace(noisy_path, enhan_path)

            # print('clean_wav_path =', clean_wav_path)
            # print('noisy_wav_path =', noisy_wav_path)
            # print('enhan_wav_path =', enhan_wav_path)
            scores.append(scoring_file(clean_wav_path, noisy_wav_path, enhan_wav_path))

    return scores # list of (file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi)

def prepare_scoring_list(test_spk_list, noise_test_list, snr_test_list, result_audio_path, result_nmaudio_path, dataset):
    test_spk_list = [str(i).zfill(2) for i in test_spk_list]

    path_list = []

    if dataset == 'AV_enh':
        for spk_num in test_spk_list:
            for noise in noise_test_list:
                for snr in snr_test_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_path, noisy_path, enhan_path)
                    path_list.extend([
                    (result_nmaudio_path + 'SP' + spk_num + '/clean/', result_nmaudio_path + 'SP' + spk_num + '/' + noise_snrdb + '/', result_audio_path + 'SP' + spk_num + '/' + noise_snrdb + '/')
                    ])

    # print('path_list =', path_list)

    return path_list

def write_score(path_list, result_model_path):
    start_time = time.time()

    # print('result_model_path =', result_model_path)
    model_detail = result_model_path.rsplit('/', 2)[-2]
    # print('model_detail =', model_detail)

    scores = scoring_dir(path_list)

    count = len(scores)
    sum_noisy_pesq = 0.0
    sum_noisy_stoi = 0.0
    sum_enhan_pesq = 0.0
    sum_enhan_stoi = 0.0

    # CSV Result Output
    f = open(result_model_path + 'Results_Report[%s].csv' % model_detail, 'w')
    w = csv.writer(f)
    w.writerow(('File_Name', 'Noisy_PESQ', 'Enhan_PESQ', 'Noisy_STOI', 'Enhan_STOI'))

    for score in scores:
        file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi = score

        sum_noisy_pesq += noisy_pesq
        sum_noisy_stoi += noisy_stoi
        sum_enhan_pesq += enhan_pesq
        sum_enhan_stoi += enhan_stoi

        w.writerow((file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi))

    mean_noisy_pesq = sum_noisy_pesq / count
    mean_noisy_stoi = sum_noisy_stoi / count
    mean_enhan_pesq = sum_enhan_pesq / count
    mean_enhan_stoi = sum_enhan_stoi / count

    print()
    print('mean_noisy_pesq = %5.3f, mean_noisy_stoi = %5.3f' % (mean_noisy_pesq, mean_noisy_stoi))
    print('mean_enhan_pesq = %5.3f, mean_enhan_stoi = %5.3f' % (mean_enhan_pesq, mean_enhan_stoi))
    print()

    w.writerow(())
    w.writerow(('total mean', mean_noisy_pesq, mean_enhan_pesq, mean_noisy_stoi, mean_enhan_stoi))
    f.close()

    # remove the by-product created by the PESQ execute file
    subprocess.call(['rm', '_pesq_itu_results.txt'])
    subprocess.call(['rm', '_pesq_results.txt'])

    end_time = time.time()

    score_time = cal_time(start_time, end_time)
    print('Scoring complete.\n')

    return score_time
