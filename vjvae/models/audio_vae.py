from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from vjvae.dataset.nsynth import NSynthDataset


AudioVAEHParams = namedtuple(
        'n_encoder_layers',
        'n_encoder_channels',
        'encoder_filter_size',
        'encoder_strides',
        'n_fc_layers',
        'n_fc_channels',
        'latent_dim',
        'n_decoder_layers',
        'n_decoder_channels',
        'decoder_filter_size',
        'decoder_strides'
        )

def get_audio_vae_h_params(
        n_conv_layers=3,
        n_conv_channels=[32, 16, 8],
        conv_filter_size=[1, 3, 3],
        conv_strides=[1, 2, 2],
        n_fc_layers=1,
        n_fc_channels=[256],
        latent_dim=32,
        ):
    assert n_conv_layers == len(n_conv_channels), 'n_conv_layers and len(n_conv_layers) need to be equal'
    assert n_fc_layers == len(n_fc_channels), 'n_fc_layers and len(n_fc_layers) need to be equal'

    config = AudioVAEConfig(
        n_encoder_layers=n_conv_layers,
        n_encoder_channels=n_conv_channels,
        encoder_filter_size=conv_filter_size,
        encoder_strides=conv_strides,
        n_fc_layers=n_fc_layers,
        n_fc_channels=n_fc_channels,
        latent_dim=latent_dim,
        n_decoder_layers=n_conv_layers,
        n_decoder_channels=n_conv_channels,
        decoder_filter_size=conv_filter_size,
        decoder_strides=conv_strides
    )
    return config

class AudioVAE(nn.Module):

    def __init__(self, hparams: AudioVAEHParams):
        super(AudioVAE, self).__init__()
        self.encoder = self.build_encoder(
                n_layers=hparams.n_encoder_layers,
                n_channels=hparams.n_encoder_channels,
                filter_size=hparams.encoder_filter_size,
                stride=hparams.encoder_strides,
                )
        self.encoder_fc = self.build_fc(
                n_layers=hparams.n_fc_layers,
                n_channels=hparams.n_fc_channels,
                )
        self.mu_fc =  self.build_fc(
                1,
                n_channels=[hparams.n_fc_channels[-1], hparams.latent_dim],
                use_batchnorm=False,
                Activation=None
                )
        self.logvar_fc = self.build_fc(
                1,
                n_channels=[hparams.n_fc_channels[-1], hparams.latent_dim],
                use_batchnorm=False,
                Activation=None
                )
        self.decoder_fc = self.build_fc(
                n_layers=hparams.n_fc_layers+1,
                n_channels=[hparams.latent_dim, *hparams.n_fc[::-1], hparams.n_fc_layers],
                use_batchnorm=False,
                )
        self.decoder = self.build_decoder(
                n_layers=hparams.n_decoder_layers,
                n_channels=hparams.n_decoder_channels,
                filter_size=hparams.decoder_filter_size,
                stride=hparams.decoder_strides,
                )

    def forward(self, mel_specgram):
        h = self.encoder(mel_specgram)
        h = self.encoder_fc(h)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)

        h = self.decoder_fc(latent)
        recon = self.decoder(h)
        return recon

    def build_encoder(
            self,
            n_layers=3,
            n_channels=[64, 32, 16, 8],
            filter_size=[1, 3, 3],
            stride=[1, 2, 2]
            ):
        layers = []
        for i in range(n_layers):
            in_channels, out_channels = n_channels[i], n_channels[i+1]
            layers += [
                nn.Conv1d(in_channels, out_channels, filter_size[i], stride[i]),
                nn.BatchNorm1d(out_channels),
                nn.Tanh()
            ]

        return nn.Sequential(*layers)

    def build_fc(
            self,
            n_layers,
            n_channels,
            use_batchnorm=True,
            Activation=nn.Tanh,
            ):
        layers = []
        for i in range(n_layers):
            layer = [nn.Linear(n_channels[i], n_channels[i+1])]
            if use_batchnorm:
                layer.append(nn.BatchNorm1d(n_channels[i+1]))

            if Activation is not None:
                layer.append(Activation())
            layers += layer

        return nn.Sequential(*layers)

    def build_decoder(
            self,
            n_layers=3,
            n_channels=[64, 32, 16, 8],
            filter_size=[1, 3, 3],
            strides=[1, 2, 2]
            ):
        layers = []
        for i in range(n_layers-1):
            in_channels, out_channels = n_channels[i], n_channels[i+1]
            layers += [
                nn.ConvTranspose1d(in_channels, out_channels, filter_size[i], stride[i]),
                nn.BatchNorm1d(out_channels),
                nn.Tanh()
            ]
        layers += [
                nn.ConvTranspose1d(n_channels[-2], n_channels[-1], filter_size[-1], stride[-1]),
                nn.Tanh()
        ]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    config = get_audio_vae_config()
    dataset = NSynthDataset(data_dir='./data/examples')
    audio_vae = AudioVAE(config)
