dataset_path = '../data/'

train_spk_list = [1]
val_spk_list = [1]
test_spk_list = [5]

noise_train_list = ['engine']
noise_val_list = ['engine']
noise_test_list = ['engine']

snr_train_list = [0]
snr_val_list = [0]
snr_test_list = [-4]

def prepare_train_path_list(train_spk_list, noise_train_list, snr_train_list, dataset):
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in train_spk_list]

    train_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:
            for noise in noise_train_list:
                for snr in snr_train_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_train_path, noisy_train_path, lip_path)
                    train_path_list.extend([
                    (spk_path + 'audio_stftpt/clean/train/', spk_path + 'audio_stftpt/noisy_enh/train/' + noise_snrdb + '/', spk_path + 'img_autoencoded_lip/')
                    ])

    return train_path_list

def prepare_val_path_list(val_spk_list, noise_val_list, snr_val_list, dataset):
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in val_spk_list]

    val_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:
            for noise in noise_val_list:
                for snr in snr_val_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_val_path, noisy_val_path, lip_path)
                    val_path_list.extend([
                    (spk_path + 'audio_stftpt/clean/val/', spk_path + 'audio_stftpt/noisy_enh/val/' + noise_snrdb + '/', spk_path + 'img_autoencoded_lip/')
                    ])

    return val_path_list

def prepare_test_path_list(test_spk_list, noise_test_list, snr_test_list, dataset):
    test_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in test_spk_list]

    test_path_list = []
    clean_path_list = []

    if dataset == 'AV_enh':
        for spk_path in test_spk_path_list:
            for noise in noise_test_list:
                for snr in snr_test_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # ('', noisy_test_path, lip_path)
                    test_path_list.extend([
                    ('', spk_path + 'audio_stftpt/noisy_enh/test/' + noise_snrdb + '/', spk_path + 'img_autoencoded_lip/')
                    ])

            # ('', clean_test_path, lip_path)
            clean_path_list.extend([
            ('', spk_path + 'audio_stftpt/clean/test/', spk_path + 'img_autoencoded_lip/')
            ])

    return test_path_list, clean_path_list

def prepare_path_list(train_spk_list, val_spk_list, test_spk_list, noise_train_list, noise_val_list, noise_test_list, snr_train_list, snr_val_list, snr_test_list, dataset):
    train_path_list = prepare_train_path_list(train_spk_list, noise_train_list, snr_train_list, dataset)
    val_path_list = prepare_val_path_list(val_spk_list, noise_val_list, snr_val_list, dataset)
    test_path_list, clean_path_list = prepare_test_path_list(test_spk_list, noise_test_list, snr_test_list, dataset)

    return train_path_list, val_path_list, test_path_list, clean_path_list
