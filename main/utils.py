import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from visdom import Visdom
import time
import numpy as np

num_workers = 8
pin_memory = False

def model_detail_string(av, model_name, epochs, lr, train_batch_size, loss_coefficient=0):
    if av:
        model_detail = '%s_e(%d)lr(%.0e)bs(%d)lc(%.0e)' % (model_name, epochs, lr, train_batch_size, loss_coefficient)
        model_detail = model_detail.replace('0e+00', '0')
    else:
        model_detail = '%s_e(%d)lr(%.0e)bs(%d)' % (model_name, epochs, lr, train_batch_size)

    return model_detail

def cal_time(start_time, end_time):
    s = end_time - start_time
    m = s // 60
    h = m // 60
    m = m % 60
    d = h // 24
    h = h % 24

    if s >= 30:
        m += 1

    return d, h, m

# Section of functions for DataLoader
def collate_fn(batch):
    file_name, frame_num, clean_or_phase, noisy, noisy_mean, noisy_std, lip = list(zip(*batch))

    min_frame_num = min(frame_num)
    indices = torch.arange(min_frame_num)

    trim_clean_or_phase = True if isinstance(clean_or_phase[0], torch.Tensor) else False
    # no self.trim_noisy because it is always true (data_generator output noisy in all mode)
    trim_noisy_mean_std = True if isinstance(noisy_mean[0], torch.Tensor) else False
    trim_lip = True if isinstance(lip[0], torch.Tensor) else False

    clean_or_phase = list(clean_or_phase)
    noisy = list(noisy)
    lip = list(lip)

    # trimming each data in a batch to the length of min_frame_num
    for i, _ in enumerate(noisy):
        if trim_clean_or_phase:
            clean_or_phase[i] = torch.index_select(clean_or_phase[i], dim=-1, index=indices)
        
        noisy[i] = torch.index_select(noisy[i], dim=-1, index=indices)
        
        if trim_lip:
            lip[i] = torch.index_select(lip[i], dim=-1, index=indices)

    clean_or_phase = torch.stack(clean_or_phase, 0) if trim_clean_or_phase else None
    noisy = torch.stack(noisy, 0)
    lip = torch.stack(lip, 0) if trim_lip else None

    if trim_noisy_mean_std:
        noisy_mean = torch.stack(noisy_mean, 0)
        noisy_std = torch.stack(noisy_std, 0)
    else:
        None

    return file_name, min_frame_num, clean_or_phase, noisy, noisy_mean, noisy_std, lip

# Section of training
def train(device, model, train_dataset, val_dataset, fromepoch, epochs, train_batch_size, frame_seq, loss_coefficient, criterion, optimizer, result_model_path):
    model.train()
    model_name = model.__class__.__name__
    lr = optimizer.defaults['lr']

    av = train_dataset.av

    if av:
        model_detail = model_detail_string(av, model_name, epochs, lr, train_batch_size, loss_coefficient)
    else:
        model_detail = model_detail_string(av, model_name, epochs, lr, train_batch_size)

    loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    idx_middle_frame = (frame_seq - 1) // 2
    patience = 10 # how many epochs training will automatically stop if val_loss not improves

    print('Visdom activating.')
    viz = Visdom(env='%s' % model_detail)
    print()

    start_time = time.time()
    print(time.asctime(time.localtime(time.time())) + ' Start training ' + model_detail + '...')
    print()

    for epoch in range(fromepoch, epochs):
        running_loss = 0.0

        if av:
            running_a_loss = 0.0
            running_v_loss = 0.0

        optimizer.zero_grad()

        with tqdm(total=total_step, desc='epoch: %2d/%2d, train_loss: %7.4f' % (epoch + 1, epochs, running_loss),
                  dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt
                file_name, total_frames, clean, noisy, _, _, lip = batch
                pred_total_frames = total_frames - frame_seq + 1

                clean = clean.to(device, non_blocking=True)
                noisy = noisy.to(device, non_blocking=True)

                if av:
                    lip = lip.to(device, non_blocking=True)

                # scanning data along frame
                for i in range(pred_total_frames):
                    if av:
                        frames_noisy = noisy[..., i:i+frame_seq]
                        frames_lip = lip[..., i:i+frame_seq]

                        audio_target = clean[..., i+idx_middle_frame]
                        visual_target = lip[..., i+idx_middle_frame]

                        pred_frame_noisy, pred_frame_lip = model(frames_noisy, frames_lip)

                        noisy_loss = criterion(pred_frame_noisy, audio_target)
                        lip_loss = criterion(pred_frame_lip, visual_target)
                        loss = noisy_loss + loss_coefficient * lip_loss

                    else:
                        frames_noisy = noisy[..., i:i+frame_seq]

                        audio_target = clean[..., i+idx_middle_frame]

                        pred_frame_noisy = model(frames_noisy)

                        loss = criterion(pred_frame_noisy, audio_target)

                    loss /= pred_total_frames
                    loss.backward()
                    running_loss += loss.item() / total_step

                    if av:
                        running_a_loss += noisy_loss.item() / pred_total_frames / total_step
                        running_v_loss += lip_loss.item() / pred_total_frames / total_step

                optimizer.step()
                optimizer.zero_grad()

                t.set_description('epoch: %2d/%2d, train_loss: %7.4f' % (epoch + 1, epochs, running_loss))
                t.update()

        if av:
            val_loss, val_a_loss, val_v_loss = val(device, model, val_dataset, train_batch_size, frame_seq, loss_coefficient, criterion)

            viz.line(
                np.column_stack((running_loss, running_a_loss, running_v_loss, val_loss, val_a_loss, val_v_loss)), np.column_stack((epoch + 1, epoch + 1, epoch + 1, epoch + 1, epoch + 1, epoch + 1)), win='loss', update='append',
                opts=dict(title='loss', xlabel='epoch', ylabel='loss', showlegend=True, legend=['training', 'training_audio', 'training_visual', 'val', 'val_audio', 'val_visual'])
            )

        else:
            val_loss = val(device, model, val_dataset, train_batch_size, frame_seq, loss_coefficient, criterion)

            viz.line(
                np.column_stack((running_loss, val_loss)), np.column_stack((epoch + 1, epoch + 1)), win='loss', update='append',
                opts=dict(title='loss', xlabel='epoch', ylabel='loss', showlegend=True, legend=['training', 'val'])
            )

        ### save best model ###
        # init
        if epoch == fromepoch:
            running_patience = patience # running_patience init
            best_loss = val_loss
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': running_loss,
                        'val_loss': val_loss}, result_model_path + 'best_model[%s].tar' % model_detail)

        # update
        if val_loss < best_loss:
            running_patience = patience # running_patience reset
            best_loss = val_loss
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': running_loss,
                        'val_loss': val_loss}, result_model_path + 'best_model[%s].tar' % model_detail)
        else:
            running_patience -= 1
            if running_patience == 0:
                print('training stopped since val_loss did not improve in the last %d epochs.' % patience)
                break

    torch.save({'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': running_loss,
                'val_loss': val_loss}, result_model_path + 'model[%s].tar' % model_detail)

    print(time.asctime(time.localtime(time.time())) + ' Training complete.')
    end_time = time.time()

    train_time = cal_time(start_time, end_time)
    print ('Model trained for %2d day %2d hr %2d min.\n' % train_time)

    return train_time

def val(device, model, val_dataset, train_batch_size, frame_seq, loss_coefficient, criterion):
    model.eval()

    av = val_dataset.av
    loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    idx_middle_frame = (frame_seq - 1) // 2

    val_loss = 0.0

    if av:
        val_a_loss = 0.0
        val_v_loss = 0.0

    with torch.no_grad():
        with tqdm(total=total_step, desc='                val_loss: %7.4f' % val_loss,
                  dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt
                file_name, total_frames, clean, noisy, _, _, lip = batch
                pred_total_frames = total_frames - frame_seq + 1

                clean = clean.to(device, non_blocking=True)
                noisy = noisy.to(device, non_blocking=True)

                if av:
                    lip = lip.to(device, non_blocking=True)

                # scanning data along frame
                for i in range(pred_total_frames):
                    if av:
                        frames_noisy = noisy[..., i:i+frame_seq]
                        frames_lip = lip[..., i:i+frame_seq]

                        audio_target = clean[..., i+idx_middle_frame]
                        visual_target = lip[..., i+idx_middle_frame]

                        pred_frame_noisy, pred_frame_lip = model(frames_noisy, frames_lip)

                        noisy_loss = criterion(pred_frame_noisy, audio_target)
                        lip_loss = criterion(pred_frame_lip, visual_target)
                        loss = noisy_loss + loss_coefficient * lip_loss

                    else:
                        frames_noisy = noisy[..., i:i+frame_seq]

                        audio_target = clean[..., i+idx_middle_frame]

                        pred_frame_noisy = model(frames_noisy)

                        loss = criterion(pred_frame_noisy, audio_target)

                    loss /= pred_total_frames
                    val_loss += loss.item() / total_step

                    if av:
                        val_a_loss += noisy_loss.item() / pred_total_frames / total_step
                        val_v_loss += lip_loss.item() / pred_total_frames / total_step

                t.set_description('                val_loss: %7.4f' % val_loss) # space for alignment of tqdm
                t.update()

    model.train()

    if av:
        return val_loss, val_a_loss, val_v_loss
    else:
        return val_loss

# Section of testing
def test(device, model, dataset, test_batch_size, frame_seq, result_audio_path, result_imgpt_path):
    model.eval()
    model_name = model.__class__.__name__

    mode = dataset.mode
    av = dataset.av
    loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    idx_middle_frame = (frame_seq - 1) // 2

    if mode == 'testing':
        start_time = time.time()
        print(time.asctime(time.localtime(time.time())) + ' Start testing ' + model_name + '...')
        print()

        if av:
            print('Saving result wav to \'' + result_audio_path + '\'\n' + \
                  '   and result lip to \'' + result_imgpt_path + '\'...')
        else:
            print('Saving result wav to \'' + result_audio_path + '\'...')

    elif mode == 'no_model':
        if av:
            print('Saving ' + dataset.name + ' no model wav to \'' + result_audio_path + '\'\n' + \
                  '   and ' + dataset.name + ' no model lip to \'' + result_imgpt_path + '\'...')
        else:
            print('Saving ' + dataset.name + ' no model wav to \'' + result_audio_path + '\'...')

    with torch.no_grad():
        with tqdm(total=total_step, dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                if mode == 'no_model':
                    # data structure: [0] file_name
                    #                 [1] frame_num
                    #                 [2] spec_clean or phase_noisy
                    #                 [3] nor_spec_noisy
                    #                 [4] spec_noisy_mean
                    #                 [5] spec_noisy_std
                    #                 [6] lippt
                    file_names, total_frames, phases, noisys, noisys_means, noisys_stds, lips = batch
                else:
                    file_names, total_frames, phases, noisys, _, _, lips = batch

                pred_total_frames = total_frames - frame_seq + 1

                noisys_shape = list(noisys.shape)
                noisys_shape[-1] = pred_total_frames
                noisys = noisys.to(device, non_blocking=True)
                pred_noisys = torch.zeros(noisys_shape).to(device, non_blocking=True)

                if av:
                    lips_shape = list(lips.shape)
                    lips_shape[-1] = pred_total_frames
                    lips = lips.to(device, non_blocking=True)
                    pred_lips = torch.zeros(lips_shape).to(device, non_blocking=True)

                for i in range(pred_total_frames):
                    if av:
                        frames_noisys = noisys[..., i:i+frame_seq]
                        frames_lips = lips[..., i:i+frame_seq]

                        if mode == 'testing':
                            pred_frame_noisys, pred_frame_lips = model(frames_noisys, frames_lips)
                        elif mode == 'no_model':
                            pred_frame_noisys, pred_frame_lips = frames_noisys[..., idx_middle_frame], frames_lips[..., idx_middle_frame]

                    else:
                        frames_noisys = noisys[..., i:i+frame_seq]

                        if mode == 'testing':
                            pred_frame_noisys = model(frames_noisys)
                        elif mode == 'no_model':
                            pred_frame_noisys = frames_noisys[..., idx_middle_frame]

                    pred_noisys[..., i] = pred_frame_noisys

                    if av:
                        pred_lips[..., i] = pred_frame_lips

                if mode == 'no_model':
                    noisys_means = noisys_means.to(device, non_blocking=True)
                    noisys_stds = noisys_stds.to(device, non_blocking=True) - 1e-12
                    pred_noisys = (pred_noisys * noisys_stds) + noisys_means

                # inverse log1p
                pred_noisys = 10 ** pred_noisys
                pred_noisys = pred_noisys - 1

                # use predicted spectrogram and phase of original wavform to calculate stft
                # recreate predicted waveform

                phases = phases.to(device, non_blocking=True)
                phases = phases[..., idx_middle_frame:idx_middle_frame+pred_total_frames]
                stfts = (pred_noisys - 1e-12) * phases
                stfts = stfts.permute(0, 2, 3, 1)

                wavs = torchaudio.functional.istft(stfts, n_fft=512, hop_length=320,
                                            win_length=512, window=torch.hann_window(512).to(stfts.device),
                                            center=True, pad_mode='reflect', normalized=False,
                                            onesided=True, length=None)

                wavs = wavs.cpu()

                if av:
                    outputs = zip(file_names, wavs, pred_lips)
                else:
                    outputs = zip(file_names, wavs)

                for output in outputs:
                    if av:
                        file_name, wav, pred_lip = output
                    else:
                        file_name, wav = output

                    wav /= torch.max(torch.abs(wav))
                    wav /= 8

                    file_name = file_name.split('__')
                    noise_type = 'clean' if file_name[1] == 'test' else file_name[1]
                    file_name = file_name[0]
                    spk_name = file_name.split('_')[0]

                    result_wav_dir_path = result_audio_path + spk_name + '/' + noise_type + '/'
                    result_wav_path = result_wav_dir_path + file_name + '.wav'

                    if not os.path.exists(result_wav_dir_path):
                        os.makedirs(result_wav_dir_path)

                    torchaudio.save(result_wav_path, wav, 16000, precision=16)

                    # comment out the code below to save lip model output if required
                    # the code below will save lip features after model processing for each noisy file
                    # but the input lip features of the same utterance are the same for all kinds of noise
                    # so maybe just choose one kind of noise type to complie the code below in order to save the disk space

                    # if av:
                    #     result_lippt_dir_path = result_imgpt_path + spk_name + '/' + noise_type + '/'
                    #     result_lippt_path = result_lippt_dir_path + file_name + '.pt'

                    #     if not os.path.exists(result_lippt_dir_path):
                    #         os.makedirs(result_lippt_dir_path)

                    #     # (2048, frames) -> (frames, 2048)
                    #     pred_lip = pred_lip.permute(0, 1)
                    #     torch.save(pred_lip, result_lippt_path)

                t.set_description('output wavs: %3d/%3d' % ((step + 1), total_step))
                t.update()

    if mode == 'testing':
        print()
        print(time.asctime(time.localtime(time.time())) + ' Testing complete.')
        end_time = time.time()

        test_time = cal_time(start_time, end_time)
        print ('Model tested for %2d day %2d hr %2d min.\n' % test_time)

        return test_time

    elif mode == 'no_model':
        print()
        print('Testing no model complete.\n')
