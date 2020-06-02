from torch.utils.data import  Dataset


class NSynthDataset(Dataset):

    def __init__(self, data_dir, sampling_rate=44100):
        self.wav_file_path_list = glob.glob(os.path.join(self.data_dir, '*.wav'))
        # wav file must be 30 seconds.
        # use preprocess steps
        self.wav_file_list = [
                torchaudio.load(wav_file, sampling_rate, normalization=True) for wav_file in self.wav_file_list
                ]

    def __getitem__(self, idx):
        wav_file = self.wav_file_list[idx]
        waveform, sampling_rate = wav_file[idx]
        transform = transforms.MelSpectrogram(sampling_rate)
        mel_specgram = transform(waveform)

        return mel_specgram

    def __len__(self):
        return len(self.wav_file_list)
