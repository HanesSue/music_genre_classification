import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import ffmpeg
import numpy as np
from mutagen import File
from pathlib import Path
from torch.utils.data import Dataset
import random
import os
import io


class AlbumDatasets(Dataset):
    """_A custom dataset class for loading and processing audio data.
    This class is designed to work with the `torch.utils.data.DataLoader` for
    efficient data loading and batching._
    Args:
        root (str): _The root directory where the audio files are stored._
        albums (list): _A list of dictionaries containing albums path._
        reload (bool): _Whether to reload the processed data._
        transform (callable, optional): _A function/transform to apply to the audio data._
        sample_rate (int): _The sample rate for audio processing._
        n_mels (int): _The number of Mel frequency bins for the Mel spectrogram._
    """

    def __init__(self, **kwargs):
        self.root = kwargs.get("root", str)
        self.albums = kwargs.get("albums", [])
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.n_mels = kwargs.get("n_mels", 128)
        reload = kwargs.get("reload", False)

        if not self._check_processed_exists() or reload:
            self._process()
        self._load()

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def processed_file(self):
        return os.path.join(self.processed_dir, "albums_processed.pt")

    def _check_processed_exists(self):
        return os.path.exists(self.processed_file)

    def _process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        print("processing music files...")
        self.idx = 0
        self.id = []
        self.metadata = []
        self.slices = []
        for album in self.albums:
            meta = self.__process_audio__(album, self.sample_rate, self.n_mels)
            for item in meta:
                self.metadata.append({"id": item["id"], "metadata": item["metadata"]})
                self.id.extend([item["id"] for _ in item["mel_specs"]])
                self.slices.extend(item["mel_specs"])

        save_data = {
            "id": self.id,
            "metadata": self.metadata,
            "slices": self.slices,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
        }
        torch.save(save_data, self.processed_file)
        print("processed data saved to: ", self.processed_file)
        return save_data

    def _load(self):
        data = torch.load(self.processed_file)
        self.id = data["id"]
        self.metadata = data["metadata"]
        self.slices = data["slices"]
        self.sample_rate = data["sample_rate"]
        self.n_mels = data["n_mels"]
        print("loaded processed data from: ", self.processed_file)

    def __repr__(self):
        albums = (
            f"AlbumDataset with {len(self.albums)} albums."
            if len(self.albums) > 1
            else f"AlbumDataset with {len(self.albums)} album."
        )
        songs = (
            f"Total {len(set(self.id))} songs."
            if len(set(self.id)) > 1
            else f"Total {len(set(self.id))} song."
        )
        return albums + "\n" + songs

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        mel_specs, id = self.slices[idx].unsqueeze(0), self.id[idx]
        anchor = mel_specs
        anchor_id = id
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(
                [i for i in range(len(self.id)) if self.id[i] == id]
            )
        positive = self.slices[positive_idx].unsqueeze(0)
        negative_id = random.choice(list(set(self.id) - {anchor_id}))
        negative_idx = random.choice(
            [i for i in range(len(self.id)) if self.id[i] == negative_id]
        )
        negative = self.slices[negative_idx].unsqueeze(0)

        return (anchor, positive, negative), id

    def get_metadata(self):
        return self.metadata

    def songs_length(self):
        return len(set(self.id))

    def __process_audio__(
        self, root_folder, sample_rate=16000, n_mels=96, slice_duration=10
    ):
        # Load the audio file
        music_extensions = (".flac", ".wav", ".mp3", ".m4a")
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels
        )
        meta = []
        slice_samples = slice_duration * sample_rate

        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(music_extensions):
                    file_path = Path(root) / file
                    file_path = file_path.as_posix()
                    try:
                        # metadata
                        audio_info = File(file_path, easy=True)
                        if audio_info is None:
                            print(f"无法从 {file_path} 读取音频信息")
                            continue

                        title = audio_info.tags.get("title", [None])[0]
                        artist = audio_info.tags.get("artist", [None])[0]
                        album = audio_info.tags.get("album", [None])[0]
                        genre = audio_info.tags.get("genre", [None])[0]
                        year = audio_info.tags.get("date", [None])[0]
                        duration = audio_info.info.length

                        # audio
                        if not file_path.endswith(".m4a"):
                            waveform, sr = torchaudio.load(file_path)
                        else:
                            out, _ = (
                                ffmpeg.input(file_path)
                                .output("pipe:", format="wav")
                                .run(capture_stdout=True, capture_stderr=True)
                            )
                            waveform, sr = torchaudio.load(io.BytesIO(out))
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)

                        if sr != sample_rate:
                            resampler = T.Resample(sr, sample_rate)
                            waveform = resampler(waveform)
                        if random.random() < 0.5:
                            waveform = self.__add_noise__(waveform)

                        mel_specs = self.__slice_audio__(
                            waveform, slice_samples, mel_transform
                        )

                        item = {
                            "id": self.idx,
                            "metadata": {
                                "title": title,
                                "artist": artist,
                                "album": album,
                                "genre": genre,
                                "year": year,
                                "duration": duration,
                            },
                            "mel_specs": [
                                mel_spec.squeeze(0) for mel_spec in mel_specs
                            ],
                        }
                        meta.append(item)
                        self.idx += 1
                    except Exception as e:
                        print(f"处理 {file_path} 时出错: {e}")
        return meta

    def __slice_audio__(self, waveform, slice_samples, transform=None):
        num_samples = waveform.shape[-1]
        slices = []
        num_slices = (num_samples + slice_samples - 1) // slice_samples  # 向上取整

        for i in range(num_slices):
            start = i * slice_samples
            end = start + slice_samples
            chunk = waveform[:, start:end]

            # 如果最后一片长度不足，填充零
            if chunk.shape[-1] < slice_samples:
                pad_size = slice_samples - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_size))  # 右侧补零

            mel_slice = transform(chunk)
            slices.append(mel_slice)

        return slices

    def __add_noise__(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
