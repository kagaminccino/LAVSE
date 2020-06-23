from torchsummaryX import summary
import os
import argparse
import time
from torch.optim import Adam

from build_model import *
from data_generator import AV_Dataset
from prepare_path_list import train_spk_list, val_spk_list, test_spk_list, noise_train_list, noise_val_list, noise_test_list, snr_train_list, snr_val_list, snr_test_list, prepare_path_list
from utils import model_detail_string, train, test, cal_time
from scoring import prepare_scoring_list, write_score

parser = argparse.ArgumentParser()

########## training ##########
parser.add_argument('--retrain', action='store_true', help='to train a new model or to retrain an existing model.')
parser.add_argument('--dataset', type=str, default='AV_enh', help='options: AV_enh')
parser.add_argument('--model', type=str, default='ADCNN', help='options: LAVSE')
parser.add_argument('--loss', type=str, default='MSE', help='option: MSE')
parser.add_argument('--opt', type=str, default='Adam', help='option: Adam')
parser.add_argument('--keeptrain', action='store_true', help='continue training of a trained model. remember to set --fromepoch.')
parser.add_argument('--fromepoch', type=int, default=0, help='the last epoch already trained.')
parser.add_argument('--epochs', type=int, default=500, help='the last epoch wanted to be trained.')
parser.add_argument('--train_batch_size', type=int, default=1, help='the batch size wanted to be trained.')
parser.add_argument('--frame_seq', type=int, default=5, help='the frames amount of model input.')
parser.add_argument('--loss_coefficient', type=float, default=0.001, help='loss = noisy_loss + loss_coefficient * lip_loss')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of the optimizer.')
########## testing ##########
parser.add_argument('--retest', action='store_true', help='to test or retest an existing model.')
parser.add_argument('--testnomodel', action='store_true', help='generate wav files which have the same length(frames) as test wav files for scoring.')
parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size wanted to be tested.')
########## scoring ##########
parser.add_argument('--rescore', action='store_true', help='to rescore test wavs. scoring will automatically start after testing even if this argument is not triggered.')

args = parser.parse_args()

retrain = args.retrain
dataset = args.dataset
model_name = args.model
loss_name = args.loss
opt_name = args.opt
keeptrain = args.keeptrain
fromepoch = args.fromepoch
epochs = args.epochs
train_batch_size = args.train_batch_size
frame_seq = args.frame_seq
loss_coefficient = args.loss_coefficient
lr = args.learning_rate
retest = args.retest
testnomodel = args.testnomodel
test_batch_size = args.test_batch_size
rescore = args.rescore

if __name__ == '__main__':

    # ********** starting **********
    print('\n********** starting **********\n')

    print('The %s model with dataset of %s.\n' % (model_name, dataset))

    start_time = time.time()

    # ********** check cuda **********

    gpu_amount = torch.cuda.device_count()

    print('#################################')
    print('torch.cuda.is_available() =', torch.cuda.is_available())
    # torch.cuda.is_available() reveals if any gpu can be used and if any gpu is assigned with CUDA_VISIBLE_DEVICES in bash script
    print('torch.cuda.device_count() =', gpu_amount)
    # torch.cuda.device_count() reveals how many gpus can be used, which is implicitly assigned in bash script with CUDA_VISIBLE_DEVICES
    # in bash script, CUDA_VISIBLE_DEVICES is assigned by the specific numbers of gpus, but not the amount
    print('#################################\n')

    device = os.environ['CUDA_VISIBLE_DEVICES']
    device = 'cpu' if device == '' else 'cuda'
    # setting default to 'cuda' indicates the program will use gpu:0
    # 0 here means the first gpu number assigned by CUDA_VISIBLE_DEVICES, but not the real gpu:0
    device = torch.device(device)

    # ********** model declare **********

    if model_name == 'LAVSE':
        model = LAVSE(frame_seq).to(device)
        av = True # audio-only: False, audio-visual: True
        input_size = [train_batch_size, 1, 257, frame_seq], [train_batch_size, 2048, frame_seq]

    else:
        raise NameError('custom models (with the parameters of av and input_size) should be written in main.py, \
             and the structure of the models should be written in build_model.py.')

    # criterion
    if loss_name == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise NameError('loss undefined.')

    if not av:
        loss_coefficient = 0

    # optimizer
    if opt_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        raise NameError('optimizer undefined.')

    # ********** result path **********

    result_path = '../result/'

    model_detail = model_detail_string(av, model_name, epochs, lr, train_batch_size, loss_coefficient)

    result_model_path = result_path + model_detail + '/'
    result_audio_path = result_model_path + 'audio/'
    result_imgpt_path = result_model_path + 'img_autoencoded/'

    result_nm_path = result_path + 'test_no_model%s/' % ('_AV' if av else '')
    result_nmaudio_path = result_nm_path + 'audio/'
    result_nmimgpt_path = result_nm_path + 'img_autoencoded/'

    # ********** init **********

    if retrain:
        model.apply(weights_init)

    train_path_list, val_path_list, test_path_list, clean_path_list = prepare_path_list(train_spk_list, val_spk_list, test_spk_list, noise_train_list, noise_val_list, noise_test_list, snr_train_list, snr_val_list, snr_test_list, dataset)

    # ********** training **********

    if retrain:
        print('\n********** training **********\n')

        train_dataset = AV_Dataset(name='train', data_path_list=train_path_list, mode='training', av=av)
        val_dataset = AV_Dataset(name='val', data_path_list=val_path_list, mode='validation', av=av)

        if not os.path.exists(result_model_path):
            os.makedirs(result_model_path)

        train_time = train(device, model, train_dataset, val_dataset, fromepoch, epochs, train_batch_size, frame_seq, loss_coefficient, criterion, optimizer, result_model_path)
        torch.cuda.empty_cache()

    elif keeptrain:
        print('\n********** keep training **********\n')

        train_dataset = AV_Dataset(name='train', data_path_list=train_path_list, mode='training', av=av)
        val_dataset = AV_Dataset(name='val', data_path_list=val_path_list, mode='validation', av=av)

        if not os.path.exists(result_model_path):
            os.makedirs(result_model_path)

        load_model_detail = model_detail_string(av, model_name, fromepoch, lr, train_batch_size, loss_coefficient)

        load_result_model_path = result_path + load_model_detail + '/'

        checkpoint = torch.load(load_result_model_path + 'model[%s].tar' % load_model_detail)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        fromepoch = checkpoint['epoch']

        train_time = train(device, model, train_dataset, val_dataset, fromepoch, epochs, train_batch_size, frame_seq, loss_coefficient, criterion, optimizer, result_model_path)
        torch.cuda.empty_cache()

    if retrain or keeptrain or retest:
        if type(input_size)==tuple:
            audio_input_size, visual_input_size = input_size
            audio_input_size = torch.zeros(*audio_input_size).to(device)
            visual_input_size = torch.zeros(*visual_input_size).to(device)
            summary(model, audio_input_size, visual_input_size)
        else:
            audio_input_size = torch.zeros(*input_size).to(device)
            summary(model, audio_input_size)

        print()

    # exit() # for training debug

    # ********** testing **********

    if retest:
        print('\n********** testing **********\n')

        if retrain or keeptrain:
            pass
        else:
            checkpoint = torch.load(result_model_path + 'best_model[%s].tar' % model_detail)
            model.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = AV_Dataset(name='test', data_path_list=test_path_list, mode='testing', av=av)

        if not os.path.exists(result_audio_path):
            os.makedirs(result_audio_path)
        if av:
            if not os.path.exists(result_imgpt_path):
                os.makedirs(result_imgpt_path)

        test_time = test(device, model, test_dataset, test_batch_size, frame_seq, result_audio_path, result_imgpt_path)
        torch.cuda.empty_cache()

    # ********** output clean and testing data with no model for scoring **********

    if testnomodel:
        print('\n********** output clean and testing data with no model for scoring **********\n')

        clean_dataset = AV_Dataset(name='clean', data_path_list=clean_path_list, mode='no_model', av=av)

        if not retest:
            test_dataset = AV_Dataset(name='test', data_path_list=test_path_list, mode='testing', av=av)

        test_dataset.mode = 'no_model'

        test(device, model, clean_dataset, test_batch_size, frame_seq, result_nmaudio_path, result_nmimgpt_path)
        test(device, model, test_dataset, test_batch_size, frame_seq, result_nmaudio_path, result_nmimgpt_path)
        torch.cuda.empty_cache()

    # ********** scoring **********

    if retest or rescore:
        print('\n********** scoring **********\n')

        path_list = prepare_scoring_list(test_spk_list, noise_test_list, snr_test_list, result_audio_path, result_nmaudio_path, dataset)
        score_time = write_score(path_list, result_model_path)

    # ********** ending **********

    print('\n********** ending **********\n')

    end_time = time.time()
    code_time = cal_time(start_time, end_time)

    print('The %s model with dataset of %s.' % (model_name, dataset))

    if retrain:
        print('      Trained for %2d day %2d hr %2d min.' % train_time)

    if retest:
        print('       Tested for %2d day %2d hr %2d min.' % test_time)

    if retest or rescore:
        print('       Scored for %2d day %2d hr %2d min.' % score_time)

    print('This code ran for %2d day %2d hr %2d min.\n' % code_time)
