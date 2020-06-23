import torch
import torch.nn as nn
import torch.nn.functional as F

layer_group = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM)

# Weight Initialize
def weights_init(m):
    if isinstance(m, layer_group):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('leaky_relu', 0.3))

class LAVSE(torch.nn.Module):
    # input: spectrogram, lip feature matrix extracted with pretrained autoencoder
    # input size: (batch_size, 1, 257, frame_seq), (batch_size, 2048, frame_seq)
    # output size: (batch_size, 1, 257), (batch_size, 2048)

    def __init__(self, frame_seq):
        super(LAVSE, self).__init__()
        self.frame_seq = frame_seq

        self.a_conv1 = nn.Conv2d(1, 32, (25, 5), 1, (12, 2), bias=False)
        self.a_conv1_in = nn.InstanceNorm2d(32, affine=True)

        self.a_pool2 = nn.MaxPool2d((2, 1))

        self.a_conv3 = nn.Conv2d(32, 32, (17, 5), 1, (8, 2), bias=False)
        self.a_conv3_in = nn.InstanceNorm2d(32, affine=True)

        self.a_conv4 = nn.Conv2d(32, 16, (9, 5), 1, (4, 2), bias=False)
        self.a_conv4_in = nn.InstanceNorm2d(16, affine=True)

        self.av_lstm2 = nn.LSTM(input_size=4096, hidden_size=256, num_layers=1, bias=False, batch_first=True, dropout=0, bidirectional=True)
        self.av_fc2 = nn.Linear(512 * frame_seq, 512, bias=False)
        self.av_fc2_ln = nn.LayerNorm(512, elementwise_affine=True)

        self.a_fc3 = nn.Linear(512, 1 * 257, bias=False)
        self.v_fc3 = nn.Linear(512, 2048, bias=False)

    def forward(self, noisy, lip):
        noisy = self.a_conv1(noisy)
        noisy = self.a_conv1_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = self.a_pool2(noisy)

        noisy = self.a_conv3(noisy)
        noisy = self.a_conv3_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = self.a_conv4(noisy)
        noisy = self.a_conv4_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = torch.flatten(noisy, start_dim=1, end_dim=2)
        noisy = noisy.permute(0, 2, 1) # -> (batch_size, frame_seq, 2048)

        lip = lip.permute(0, 2, 1) # -> (batch_size, frame_seq, 2048)

        x = torch.cat((noisy, lip), 2) # -> (batch_size, frame_seq, 4096)

        self.av_lstm2.flatten_parameters()
        x, (hn, cn) = self.av_lstm2(x) # -> (batch_size, frame_seq, 512)
        x = torch.flatten(x, start_dim=1) # -> (batch_size, 2560)
        x = self.av_fc2(x) # -> (batch_size, 512)
        x = self.av_fc2_ln(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        noisy = self.a_fc3(x)
        noisy = F.leaky_relu(noisy, negative_slope=0.3, inplace=True)
        noisy = noisy.view(-1, 1, 257)

        lip = self.v_fc3(x)
        lip = F.leaky_relu(lip, negative_slope=0.3, inplace=True)
        lip = lip.view(-1, 2048)

        return noisy, lip
