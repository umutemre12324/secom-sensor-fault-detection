
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_FCN(nn.Module):

    def __init__(self, input_size=586, lstm_hidden_size=128, num_classes=2):
        super(LSTM_FCN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=8,
            padding=4
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2
        )

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            lstm_hidden_size + 256,
            num_classes
        )

    def forward(self, x):

        # LSTM kolu: (batch, 586) -> (batch, 586, 1)
        x_lstm = x.unsqueeze(-1)

        _, (h_n, c_n) = self.lstm(x_lstm)
        lstm_features = h_n[-1]

        # CNN kolu: (batch, 586) -> (batch, 1, 586)
        x_cnn = x.unsqueeze(1)

        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))

        # (batch, 256, uzunluk) -> (batch, 256)
        cnn_features = self.gap(x_cnn).squeeze(-1)

        # (batch, 128) + (batch, 256) -> (batch, 384)
        combined = torch.cat(
            [lstm_features, cnn_features],
            dim=1
        )

        # (batch, 384) -> (batch, 2)
        out = self.fc(combined)

        return out
