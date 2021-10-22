import torch
import torch.nn as nn
import numpy as np
from rave.model.layers import ResidualLayerV2, ResidualLayer


class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Autoencoder(nn.Module):
    def __init__(
            self,
            input_shape,
            num_scans,
            num_types=14,
            projection=None,
            num_layer=3,  # [1, 2, 3]
            num_feature=[256, 128, 64],  # [32, 64, 128, 256, 512]
            num_hidden=32,  # [10-100??]
            dropout=0.1,  # [.1 - .5]
            num_layer_scan_classifier=2,  # [1, 2, 3]
            num_feature_scan_classifier=128,  # [32, 64, 128, 256, 512]
            dropout_scan_classifier=0.1,  # [.1 - .5]
            num_layer_type_classifier=2,  # [1, 2, 3]
            num_feature_type_classifier=128,  # [32, 64, 128, 256, 512]
            dropout_type_classifier=0.1,  # [.1 - .5]
            device="cuda"
    ):
        super().__init__()

        if type(num_feature) is int:
            num_feature = [num_feature // 2 ** i for i in range(num_layer)]

        self.input_shape = input_shape
        self.num_feature = num_feature
        self.num_hidden = num_hidden

        # SVD Initialisation
        latent_in = nn.Linear(input_shape, num_feature[0], bias=False)
        latent_out = nn.Linear(num_hidden, input_shape, bias=False)
        if projection is not None:
            proj = projection[:, :num_feature[0]]
            latent_in.weight.data = torch.tensor(
                proj, dtype=torch.float32, device=device).T
            latent_out.weight.data = torch.tensor(
                proj.T, dtype=torch.float32, device=device)
        dims = np.concatenate([num_feature, [num_hidden]])
        # Encoder
        encoder = [latent_in]
        encoder += [
            ResidualLayerV2(dims[i], dims[i + 1], dropout) for i in range(num_layer)]
        # initialize near identity:
        encoder = nn.Sequential(*encoder)
        self.encoder = nn.Sequential(encoder)

        # Decoder
        decoder = [nn.Linear(num_hidden, num_hidden)]
        decoder += [
            ResidualLayerV2(dims[-i - 1], dims[-i - 2], dropout) for i in range(num_layer)]
        decoder += [nn.Linear(num_feature[0], input_shape)]
        # initialize near identity:
        decoder[-1].weight.data *= 1e-3
        decoder[-1].bias.data *= 1e-3
        decoder = nn.Sequential(*decoder)
        self.decoder = nn.Sequential(decoder)

        # Scan Classifier
        scan_classifier = [
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_feature_scan_classifier)
        ]
        scan_classifier += [
            ResidualLayer(num_feature_scan_classifier,
                          dropout_scan_classifier) for _ in range(
                num_layer_scan_classifier)
        ]
        scan_classifier += [
            nn.Linear(num_feature_scan_classifier, num_scans),
            # nn.Softmax(dim=1)
        ]
        self.scan_classifier = nn.Sequential(*scan_classifier)

        # Type Classifier
        type_classifier = [
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_feature_type_classifier)
        ]
        type_classifier += [
            ResidualLayer(num_feature_type_classifier,
                          dropout_type_classifier) for _ in range(
                num_layer_type_classifier)
        ]
        type_classifier += [
            nn.Linear(num_feature_type_classifier, num_types),
            # nn.Softmax(dim=1)
        ]
        self.type_classifier = nn.Sequential(*type_classifier)

        self.to(device)

    def forward(self, x):
        z = self.encoder(x)
        y_scan = self.scan_classifier(z)
        y_type = self.type_classifier(z)
        x_rec = self.decoder(z)
        return z, y_scan, y_type, x_rec


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
