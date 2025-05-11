import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import pandas as pd
import torchaudio
import ast  # 用于解析字符串格式的列表


class FMADataset(Dataset):
    """_A custom dataset class for lazy loading FMA audiosets.
    This classs is desighed to work with `torch.utils.data.Dataloade` for
    efficeient data loading and batching._
    Args:
        metadata_dir (str): _The root where the meta files are stored._
        audio_dir (str): _The root where the audio files are stored._
        sample_rate (int): _The sample rate for audio processing._
        duration (int): _The duration of each audio sample._
    """

    def __init__(
        self,
        metadata_dir="fma_metadata",
        audio_dir="fma_small",
        train=True,
        validation=False,
        sample_rate=22050,
        duration=None,
    ):
        self.metadata_dir = metadata_dir
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.train = train
        self.validation = validation
        self._load_metadata()

    @property
    def subset(self):
        name = os.path.basename(self.audio_dir)
        if name in ["fma_small", "fma_medium", "fma_large", "fma_full"]:
            return name.split("_")[-1]
        raise ValueError(f"Unknown dataset size: {self.audio_dir}")
    
    # TODO
    # def _get_url(self):
    #     if self.subset == "small":
    #         return "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    #     elif self.subset == "medium":
    #         return "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    #     elif self.subset == "large":
    #         return "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
    #     elif self.subset == "full":
    #         return "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
    #     else:
    #         raise ValueError(f"Unknown dataset size: {self.subset}")
    
    # def _download(self):
    #     url = self._get_url()
    #     zip_path = os.path.join(self.audio_dir, f"{self.subset}.zip")
        

    def _load_metadata(self):
        self.tracks = pd.read_csv(
            os.path.join(self.metadata_dir, "tracks.csv"),
            header=[0, 1],
            index_col=0,
            low_memory=False,
        )
        self.tracks = self.tracks[self.tracks[("set", "subset")] == self.subset]
        if self.train:
            self.tracks = self.tracks[self.tracks[("set", "split")] == "training"]
            if self.validation:
                print("Validation is not available in training set")
        elif self.validation:
            self.tracks = self.tracks[self.tracks[("set", "split")] == "validation"]
            if self.train:
                print("Training is not available in validation set")
        else:
            self.tracks = self.tracks[self.tracks[("set", "split")] == "test"]
        self.genres = pd.read_csv(os.path.join(self.metadata_dir, "genres.csv"))
        self.genre_id_to_name = dict(zip(self.genres["genre_id"], self.genres["title"]))
        self.genre_name_to_id = dict(zip(self.genres["title"], self.genres["genre_id"]))

    def _get_audio_path(self, track_id):
        tid_str = str(track_id).zfill(6)
        return os.path.join(self.audio_dir, tid_str[:3], f"{tid_str}.mp3")

    def _get_genres(self, track_id):
        raw = self.tracks.loc[track_id, ("track", "genres_all")]
        if pd.isna(raw):
            return []
        genre_ids = ast.literal_eval(raw)  # 将字符串形式的列表转为真正的 list
        return [self.genre_id_to_name.get(gid, f"unknown-{gid}") for gid in genre_ids]

    def _get_top_genre(self, track_id):
        name = self.tracks.loc[track_id, ("track", "genre_top")]
        if pd.isna(name):
            return None
        genre_id = self.genre_name_to_id.get(name, None)
        return genre_id, name

    def _get_one_hot(self, genres):
        one_hot = torch.zeros(len(self.genres), dtype=torch.float32)
        for genre in genres:
            if genre in self.genre_id_to_name.values():
                index = list(self.genre_id_to_name.values()).index(genre)
                one_hot[index] = 1.0
        return one_hot

    def get_track_info(self, track_id):
        path = self._get_audio_path(track_id)
        genres = self._get_genres(track_id)
        _, genre_top = self._get_top_genre(track_id)

        return {"id": track_id, "genres": genres, "genre_top": genre_top, "path": path}

    def get_genre_tree(self):
        tree = {}
        genre_df = self.genres.set_index("genre_id")

        def build_tree(genre_id):
            path = []
            while not pd.isna(genre_id):
                path.append(int(genre_id))
                parent = genre_df.loc[genre_id]["parent"]
                if pd.isna(parent) or parent not in genre_df.index:
                    break
                genre_id = int(parent)
            return list(reversed(path))

        for gid, row in genre_df.iterrows():
            path = build_tree(gid)
            tree[gid] = {
                "name": row["title"],
                "parent": int(row["parent"]) if not pd.isna(row["parent"]) else None,
                "path": path,
                "level": len(path),
            }
        return tree

    def load_audio(self, track_id):
        path = self._get_audio_path(track_id)
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            # print(f"Failed to load {path}: {e}")
            return None
        duration = waveform.size(1) / sr
        if duration < 1.0:  # 设定最小时长为 1 秒
            return None  # 返回 None 跳过该文件

        if self.sample_rate and sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.duration:
            max_length = int(self.sample_rate * self.duration)
            current_length = waveform.size(1)
            if current_length < max_length:
                pad = torch.zeros((1, max_length - current_length))
                waveform = torch.cat([waveform, pad], dim=1)
            elif current_length > max_length:
                waveform = waveform[:, :max_length]
        return waveform

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_id = self.tracks.index[idx]
        try:
            results = self._get_tuple(track_id)
            if results is None:
                return None
            waveform, genres, genre_id = results
            genres = self._get_one_hot(genres)
            return waveform, genres, genre_id
        except Exception as e:
            # print(f"Failed to load track {track_id}: {e}")
            return None

    def _get_tuple(self, track_id):
        genres = self._get_genres(track_id)
        genre_top = self._get_top_genre(track_id)
        if not genre_top:
            return None
        genre_id, genre_top = genre_top

        try:
            waveform = self.load_audio(track_id)
            if waveform is None:
                return None
            return (waveform, genres, genre_id)
        except Exception as e:
            # print(f"Failed to load {track_id}: {e}")
            return None
