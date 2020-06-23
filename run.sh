### about the paths of the python codes

# check main/prepare_path_list.py if you want to use other datasets
# check main/scoring.py if you want to change the directories for scoring PESQ and STOI

### about the comand line below

# see main/main.py for further information about the command line

# if using gpu(s), gpu number(s) is/are assigned by CUDA_VISIBLE_DEVICES
# example: CUDA_VISIBLE_DEVICES=0,1,3
# but using only one gpu is suggested
# example: CUDA_VISIBLE_DEVICES=3
# if using cpu
# example: CUDA_VISIBLE_DEVICES=''

# --model: LAVSE
#          see main/build_model.py for further information

# --loss: MSE

# --opt: Adam

# --retrain, --keeptrain, --retest, --testnomodel, --rescore: True if any input
# --retrain: if you want to train a new model, you have to activate visdom before training (for loss visualization)
# --testnomodel: this outputs frame-dropped reference for scoring PESQ and STOI (since we use five frames to predict one frame, the enhanced audio has less frames)
# --rescore: for only scoring. if --retest is True, scoring will automatically happen even if --rescore is not True.

# --loss_coefficient: the weight of lip_loss, denoted by lc in results
#                     loss = noisy_loss + loss_coefficient * lip_loss
#                     if audio-only models are used, loss_coefficient will be set to 0 no matter what the loss_coefficient is given by the command line

cd main

CUDA_VISIBLE_DEVICES=0 python main.py --retrain \
                              --dataset AV_enh \
                              --model LAVSE --loss MSE --opt Adam \
                              --epochs 42 --train_batch_size 32 \
                              --frame_seq 5 --loss_coefficient 0.001 --learning_rate 5e-5 \
                              --retest \
                              --testnomodel \
                              --test_batch_size 1
