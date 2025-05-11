from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F


class ResNet18Simple(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Simple, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34Classifier, self).__init__()
        self.resnet = models.resnet34(weights="ResNet34_Weights.DEFAULT")

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)
        return x


class VGG19_BN(nn.Module):
    def __init__(self, num_classes=10, full_train=False):
        super(VGG19_BN, self).__init__()
        self.full_train = full_train
        self.vgg = models.vgg19_bn(weights="VGG19_BN_Weights.DEFAULT")
        self.vgg.classifier[-1] = nn.Linear(4096, num_classes)
        if not self.full_train:
            for param in self.vgg.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.vgg(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: (B, T, D)
        attn_scores = self.attention(x)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        weighted = x * attn_weights  # (B, T, D)
        return weighted.sum(dim=1)  # (B, D)


class CRNN(nn.Module):
    def __init__(self, num_classes=10, freq=128):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.rnn = nn.LSTM(
            input_size=128, hidden_size=64, batch_first=True, bidirectional=True
        )
        self.rnn_drop = nn.Dropout(0.3)
        self.atten_pooling = AttentionPooling(input_dim=128, hidden_dim=128)
        self.pre_drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, f, t)
        x = self.conv(x)  # -> (B, 128, 1, t')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # -> (B, t', C, 1)
        x = x.reshape(B, T, C * F)  # -> (B, t', input_size)

        x, _ = self.rnn(x)  # -> (B, t', 128)
        x = self.rnn_drop(x)
        x = self.atten_pooling(x)  # -> (B, 128)
        x = self.pre_drop(x)
        x = self.fc(x)  # -> (B, num_classes)
        return x


class TemporalCNN_old(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size=3, num_filters=32):
        super(TemporalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_channels, num_filters, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                num_filters, num_filters * 2, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                num_filters * 2, num_filters * 4, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(num_filters * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalCNN(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size=3, num_filters=32):
        super(TemporalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_channels, num_filters, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                num_filters, num_filters * 2, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters * 2),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                num_filters * 2, num_filters * 4, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(num_filters * 4),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(num_filters * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FeatureBranch(nn.Module):
    def __init__(self, input_channels=1, cnn_out=32, lstm_hidden=64):
        super(FeatureBranch, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                input_channels,
                cnn_out,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(cnn_out),
            nn.ReLU(),
            nn.Conv2d(
                cnn_out,
                cnn_out * 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(cnn_out * 2),
            nn.ReLU(),
            nn.Conv2d(
                cnn_out * 2,
                cnn_out * 4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(cnn_out * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(B, T, C * F)
        out, _ = self.lstm(x)
        return out


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.atten = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1)
        )

    def forward(self, features):
        feats = torch.stack(features, dim=2)
        B, T, N, D = feats.shape
        feats_flat = feats.reshape(-1, D)
        scores = self.atten(feats_flat)
        scores = scores.reshape(B, T, N, 1)
        weights = F.softmax(scores, dim=2)
        fused_feature = (feats * weights).sum(dim=2)
        return fused_feature


class MultibranchLSTM(nn.Module):
    def __init__(self, num_classes=8, resized=True):
        super(MultibranchLSTM, self).__init__()
        self.resized = resized
        self.branch_mel = FeatureBranch(input_channels=1, cnn_out=32, lstm_hidden=64)
        self.branch_cqt = FeatureBranch(input_channels=1, cnn_out=32, lstm_hidden=64)
        self.branch_chroma = FeatureBranch(input_channels=1, cnn_out=32, lstm_hidden=64)
        self.fusion = AttentionFusion(feature_dim=64 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if self.resized:
            mel = x[:, 0:1, :, :]
            cqt = x[:, 1:2, :, :]
            chroma = x[:, 2:3, :, :]
        else:
            mel, cqt, chroma = x
        mel_features = self.branch_mel(mel)
        cqt_features = self.branch_cqt(cqt)
        chroma_features = self.branch_chroma(chroma)
        features = [mel_features, cqt_features, chroma_features]
        fused_feature = self.fusion(features)  # (batch, seq_len, hidden*2)
        fused_feature = fused_feature.mean(dim=1)
        x = self.classifier(fused_feature)
        return x


class CRNNwithResNet(nn.Module):
    def __init__(self, num_classes=8, resnet="resnet18"):
        super(CRNNwithResNet, self).__init__()
        self.input_adaptor = nn.Conv2d(1, 3, kernel_size=(1, 1))
        if resnet == "resnet18":
            resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        elif resnet == "resnet34":
            resnet = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, None))
        resnet_out_channel = resnet.layer4[-1].conv1.out_channels

        self.rnn = nn.LSTM(
            input_size=resnet_out_channel,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_drop = nn.Dropout(0.3)
        self.atten_pooling = AttentionPooling(input_dim=64, hidden_dim=64)
        self.pre_drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(64, num_classes),
        )  # 64 * 2 双向

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.input_adaptor(x)

        x = self.feature_extractor(x)  # x: (B, C, 1, t)
        x = self.global_avg_pool2d(x)
        x = x.permute(0, 3, 1, 2)  # -> (B, t, C, 1)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)  # -> (B, t', input_size)

        x, _ = self.rnn(x)  # -> (B, t', 128)
        x = self.rnn_drop(x)
        x = self.atten_pooling(x)  # -> (B, 128)
        x = self.pre_drop(x)
        x = self.fc(x)  # -> (B, num_classes)
        return x


class DeepCRNN(nn.Module):
    def __init__(self, num_classes=8):
        super(DeepCRNN, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            nn.Dropout(0.1),
        )

        self.rnn1 = nn.GRU(128, 32, batch_first=True, bidirectional=False)
        self.rnn2 = nn.GRU(32, 32, batch_first=True, bidirectional=False)

        self.drop_final = nn.Dropout(0.3)

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.pad(x, (37, 37), mode="constant", value=0)
        x = self.bn0(x)
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.drop_final(x[:, -1, :])
        x = self.fc(x)
        return x
