from collections import namedtuple


Config = namedtuple(
        'data_dir',
        )

def get_audio_vae_config(data_path):
    return Config(data_dir=data_dir)
