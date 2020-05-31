import logging, multiprocessing, os

import tensorflow.compat.v1 as tf

from magenta.models.music_vae import data, configs
from magenta.music.protobuf import music_pb2

from dataset import Dataset

class Midi(Dataset):

    '''Dataloader Superclass'''
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = [os.path.join(data_path, 'midi/', cur_path) for cur_path in os.listdir(os.path.join(data_path, 'midi/'))]

    def _load_train_audio(self, path):
        reader = tf.python_io.tf_record_iterator(path['audio'])
        audios = []
        for serialized_sequence in reader:
            audio = music_pb2.NoteSequence.FromString(serialized_sequence)
            audios.append(audio)

        return {'audios': audios}

    def _load_test_audio(self, path):
        reader = tf.python_io.tf_record_iterator(path['audios'])
        audios = []
        for serialized_sequence in reader:
            audio = music_pb2.NoteSequence.FromString(serialized_sequence)
            audios.append(audio)
        return {'audios': audios}

    def _load_train_data(self, path):
        reader = tf.python_io.tf_record_iterator(path['audios'])
        audios = []
        for serialized_sequence in reader:
            audio = music_pb2.NoteSequence.FromString(serialized_sequence)
            audios.append(audio)
        return {'audios': audios}

    def _load_test_data(self, path):
        reader = tf.python_io.tf_record_iterator(path['audios'])
        audios = []
        for serialized_sequence in reader:
            audio = music_pb2.NoteSequence.FromString(serialized_sequence)
            audios.append(audio)
        return {'audios': audios}

    def split_train_data(self):
        split_idx = int(len(self.data)*.8)
        train_audios = self.data[:split_idx]
        valid_audios = self.data[split_idx:]
        logging.info("[MIDI] Split data into %d training and %d validation audios." % (len(train_audios), len(valid_audios)))
        return train_audios, valid_audios

    def get_iterator(self, batch_size):
        audio_iters = []

        for data in self.data:
            audio = note_sequence_io.note_sequence_record_iterator(path)
            audio_iters.append(audio)

        dataset = itertool.chain(*audio_iters)
        iterator = dataset.make_initializable_iterator()
        return iterator


    def get_audio_iterator(self, batch_size):
        audio_iters = []

        for data in self.data:
            audio = note_sequence_io.note_sequence_record_iterator(path)
            audio_iters.append(audio)

        dataset = itertool.chain(*audio_iters)
        iterator = dataset.make_initializable_iterator()
        return iterator


    def get_train_iterators(self, batch_size, buffer_size=1000):
        # train_audios, valid_audios = self.split_train_data()
        # # construct training dataset
        # train_paths = tf.data.Dataset.from_tensor_slices({'audios': train_audios})
        # train_dataset = train_paths.map(self._load_train_data, num_parallel_calls=multiprocessing.cpu_count())
        # train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        # train_iterator = train_dataset.make_initializable_iterator()
        # # construct validation dataset

        train_iterator = data.get_dataset(
                self.config,
                tf_file_reader=tf.data.TFRecordDataset,
                is_training=True,
                )
        valid_iterator = data.get_dataset(
                self.config,
                tf_file_reader=tf.data.TFRecordDataset,
                is_training=False,
                )

        return train_iterator, valid_iterator


    def get_train_audio_iterators(self, batch_size, buffer_size=1000):
        # train_audios, valid_audios = self.split_train_data()
        # # construct training dataset
        # train_paths = tf.data.Dataset.from_tensor_slices(train_audios)
        # train_dataset = train_paths.map(self._load_train_audio, num_parallel_calls=multiprocessing.cpu_count())
        # train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        # train_iterator = train_dataset.make_initializable_iterator()
        # # construct validation dataset
        # valid_paths = tf.data.Dataset.from_tensor_slices(valid_audios)
        # valid_dataset = valid_paths.map(self._load_test_audio, num_parallel_calls=multiprocessing.cpu_count())
        # valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
        # valid_iterator = valid_dataset.make_initializable_iterator()
        train_iterator = data.get_dataset(
                self.config,
                tf_file_reader=tf.data.TFRecordDataset,
                is_training=True,
                )
        valid_iterator = data.get_dataset(
                self.config,
                tf_file_reader=tf.data.TFRecordDataset,
                is_training=False,
                )

        return train_iterator, valid_iterator

if __name__ == "__main__":
    dataset = Midi(data_path='/data2/yamad/vjvae')
    train_iterator, valid_iterator = dataset.get_train_iterators(batch_size=64)
    train_audio_iterator, valid_iterator = dataset.get_train_audio_iterators(64)

    with tf.Session() as sess:
        next_op = next(train_iterator)
        batch = sess.run(next_op)
