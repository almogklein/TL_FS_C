# import ast_models as models
# import argparse
# import pickle


import os
import csv
import json
import torch
import random
import torchaudio
import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset


class ESC_TL_Dataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, class_map, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.cla_map = class_map
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        
        self.index_dict = self.make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        
    def make_index_dict(self, label_csv):
        index_lookup = {}
        
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                index_lookup[row['mid']] = row['index']
        return index_lookup

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(label_str.split('j')[1])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
    
class ESC_FSL_Dataset(Dataset):
    
    def __init__(self, pairs_path, audio_conf):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        
        with open(pairs_path, 'r') as fp:
            pairs_json = json.load(fp)
        
        self.data = pairs_json
        
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        self.noise = self.audio_conf.get('noise')
        
        print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        
        # self.fbank_queue = []
        # self._precompute_fbank()
        # # if add noise for data augmentation
    
    def _precompute_fbank(self):
        for datum in self.data:
            fbank1, fbank2 = self._wav2fbank(datum[3], datum[4])
            self.fbank_queue.append([fbank1, fbank2])

    def _wav2fbank(self, filename, filename2):
   
        waveform1, sr = torchaudio.load(filename)
        waveform2, _ = torchaudio.load(filename2)

        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

        fbank1 = torchaudio.compliance.kaldi.fbank(waveform1, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        fbank2 = torchaudio.compliance.kaldi.fbank(waveform2, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames1 = fbank1.shape[0]
        n_frames2 = fbank2.shape[0]

        p1 = target_length - n_frames1
        p2 = target_length - n_frames2

        # cut and pad
        if p1 > 0 and p2 > 0:
            m1 = torch.nn.ZeroPad2d((0, 0, 0, p1))
            fbank1 = m1(fbank1)
            
            m2 = torch.nn.ZeroPad2d((0, 0, 0, p2))
            fbank2 = m2(fbank2)
        
        elif p1 > 0 and p2 < 0:
            m1 = torch.nn.ZeroPad2d((0, 0, 0, p1))
            fbank1 = m1(fbank1)
            
            fbank2 = fbank2[0:target_length, :]
        
        elif p1 < 0 and p2 > 0:
            fbank1 = fbank1[0:target_length, :]
            
            m2 = torch.nn.ZeroPad2d((0, 0, 0, p2))
            fbank2 = m2(fbank2)
            
        elif p1 < 0 and p2 < 0:
            fbank1 = fbank1[0:target_length, :]
            
            fbank2 = fbank2[0:target_length, :]

        return fbank1, fbank2 

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        
        # fbank1, fbank2 = self.fbank_queue[index]
        fbank1, fbank2 = self._wav2fbank(datum[3], datum[4])
        
        
        label_indices = int(datum[0])
        real_class = [datum[1], datum[2], datum[3], datum[4]]

        # SpecAug, not do for eval set
        if self.audio_conf.get('mode') == 'train':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
        else: 
            freqm = self.freqm
            timem = self.timem
            
        fbank1 = torch.transpose(fbank1, 0, 1)
        fbank2 = torch.transpose(fbank2, 0, 1)
        
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank1 = fbank1.unsqueeze(0)
        fbank2 = fbank2.unsqueeze(0)
        
        if self.freqm != 0:
            fbank1 = freqm(fbank1)
            fbank2 = freqm(fbank2)
            
        if self.timem != 0:
            fbank1 = timem(fbank1)
            fbank2 = timem(fbank2)
        
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank1 = fbank1.squeeze(0)
        fbank1 = torch.transpose(fbank1, 0, 1)

        fbank2 = fbank2.squeeze(0)
        fbank2 = torch.transpose(fbank2, 0, 1)
        
        # normalize the input for both training and test
        fbank1 = (fbank1 - self.norm_mean) / (self.norm_std * 2)
        fbank2 = (fbank2 - self.norm_mean) / (self.norm_std * 2)

        return fbank1, fbank2, label_indices, real_class

    def __len__(self):
        return len(self.data)

# Define a custom dataset class for the ESC-50 dataset
class ESC_TL_Dataset_wav2vec(Dataset):
  def __init__(self, audio_folder: str, metadata_file: str, label_map: dict, label_map2: dict, split: str, label_num=50):
    self.audio_folder = audio_folder
    self.metadata_file = metadata_file
    self.label_map = label_map
    self.label_map2 = label_map2
    self.split = split
    self.label_num = label_num
    self.data = self._load_metadata()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    
    audio_path = os.path.join(self.audio_folder, self.data[index]['filename'])
    
    label_indices = np.zeros(self.label_num)
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)

    label = self.data[index]['label']
    # label = self.label_map2[str(self.label_map[label])]
    label = self.label_map[label]

    label_indices[int(label)] = 1.0
    label_indices = torch.FloatTensor(label_indices)
    
    return waveform, label, label_indices

  def _load_metadata(self):
    metadata = []
    with open(self.metadata_file, 'r') as f:
      for line in f:
        filename, fold, label, category = line.strip().split(',')[:4]
        # if label in self.label_map2.keys():
        if category in self.label_map.keys():
          metadata.append({'filename': filename, 'label': category, 'fold': fold})    
    if self.split == 'train':
      metadata = [m for m in metadata if m['fold'] != '4']  # Use folds 1-4 for training
    else:
      metadata = [m for m in metadata if m['fold'] == '4']  # Use fold 5 for validation
    return metadata
