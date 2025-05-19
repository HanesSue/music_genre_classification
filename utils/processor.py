import torch
import librosa
import torch.nn.functional as F
import numpy as np


def log_mel(waveform, sample_rate=12000, hop_length=256):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform.numpy(),
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=512,
        n_mels=96,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return torch.tensor(log_mel)


def CQT(waveform, sample_rate=22050, hop_length=512):
    cqt = librosa.cqt(
        waveform.numpy(),
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=84,
        bins_per_octave=12,
    )
    cqt_mag = np.abs(cqt)
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    cqt_db = (cqt_db - cqt_db.mean()) / (cqt_db.std() + 1e-9)
    return torch.tensor(cqt_db)


def Chroma(waveform, sample_rate=22050, hop_length=512):
    y = waveform.numpy()
    if np.abs(y).max() < 1e-4:  # 可调节阈值
        return torch.zeros((12, y.shape[-1] // hop_length + 1), dtype=torch.float32)
    chroma = librosa.feature.chroma_stft(
        y=waveform.numpy(),
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=2048,
        n_chroma=12,
    )
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-9)
    return torch.tensor(chroma)


def Resize(features, target_F, chunk=False):
    if not chunk:
        resized = []
        for feat in features:
            _, f, t = feat.shape
            feat = feat.unsqueeze(0)
            feat = F.interpolate(
                feat, size=(target_F, t), mode="bilinear", align_corners=False
            )
            resized.append(feat.squeeze(0))

    else:
        resized = []
        for feat in features:
            _, _, _, t = feat.shape
            feat = feat
            feat = F.interpolate(
                feat, size=(target_F, t), mode="bilinear", align_corners=False
            )
            resized.append(feat)

    return torch.stack(resized)


def make_chunk(waveform, sample_rate=22050, chunk_time=15, time_step=7.5):
    chunk_size = int(sample_rate * chunk_time)
    step_size = int(sample_rate * time_step)
    chunks = waveform.unfold(dimension=1, size=chunk_size, step=step_size)
    return chunks


def process_chunks(waveforms, feature_fn):
    flat_waveforms = [w for chunk in waveforms for w in chunk.squeeze(0)]
    features = [feature_fn(w).unsqueeze(0) for w in flat_waveforms]
    n_chunks = waveforms[0].shape[1]

    grouped_features = [
        torch.stack(features[i : i + n_chunks])
        for i in range(0, len(features), n_chunks)
    ]

    return grouped_features


def collate_fn(batch, id_mappings, chunk=False, one_hot=False, resize=True):
    batch = [b for b in batch if b is not None]  # 过滤掉无效项
    if len(batch) == 0:
        return None, None, None  # 所有项都无效
    waveforms, genres, genre_tops = zip(*batch)
    genre_tops = [id_mappings[genre_top] for genre_top in genre_tops]
    genre_tops = torch.tensor(genre_tops)
    genres = torch.stack(genres)
    num_classes = len(id_mappings)
    if chunk:
        waveforms = [make_chunk(waveform) for waveform in waveforms]
        mel_specs = torch.stack(process_chunks(waveforms, log_mel))
        cqt = torch.stack(process_chunks(waveforms, CQT))
        chroma = torch.stack(process_chunks(waveforms, Chroma))
        if one_hot:
            genre_tops = torch.nn.functional.one_hot(
                genre_tops, num_classes=num_classes
            )
            genre_tops = genre_tops.unsqueeze(1)
            genre_tops = genre_tops.expand(-1, mel_specs.shape[1], -1).reshape(
                -1, genre_tops.shape[-1]
            )
        else:
            genre_tops = genre_tops.unsqueeze(1).repeat(1, mel_specs.shape[1]).view(-1)
        genres = genres.repeat_interleave(mel_specs.shape[1], dim=0)
        if resize:
            F_mel = mel_specs[-1].shape[2]
            F_cqt = cqt[-1].shape[2]
            F_chroma = chroma[-1].shape[2]
            F_max = max(F_mel, F_cqt, F_chroma)
            mel_specs = Resize(mel_specs, F_max, chunk)
            cqt = Resize(cqt, F_max, chunk)
            chroma = Resize(chroma, F_max, chunk)
            mel_specs = mel_specs.reshape(-1, *mel_specs.shape[2:])
            cqt = cqt.reshape(-1, *cqt.shape[2:])
            chroma = chroma.reshape(-1, *chroma.shape[2:])
            feat = torch.cat((mel_specs, cqt, chroma), dim=1)
        else:
            mel_specs = mel_specs.reshape(-1, *mel_specs.shape[2:])
            cqt = cqt.reshape(-1, *cqt.shape[2:])
            chroma = chroma.reshape(-1, *chroma.shape[2:])
            feat = (mel_specs, cqt, chroma)

    else:
        mel_specs = torch.stack(
            [log_mel(waveform[0]).unsqueeze(0) for waveform in waveforms]
        )
        cqt = torch.stack([CQT(waveform[0]).unsqueeze(0) for waveform in waveforms])
        chroma = torch.stack(
            [Chroma(waveform[0]).unsqueeze(0) for waveform in waveforms]
        )
        if one_hot:
            genre_tops = torch.nn.functional.one_hot(
                genre_tops, num_classes=num_classes
            )
            genre_tops = genre_tops.unsqueeze(1)
            genre_tops = genre_tops.expand(-1, mel_specs.shape[1], -1).reshape(
                -1, genre_tops.shape[-1]
            )
        else:
            genre_tops = genre_tops.unsqueeze(1).repeat(1, mel_specs.shape[1]).view(-1)
        if resize:
            F_mel = mel_specs[-1].shape[1]
            F_cqt = cqt[-1].shape[1]
            F_chroma = chroma[-1].shape[1]
            F_max = max(F_mel, F_cqt, F_chroma)
            mel_specs = Resize(mel_specs, F_max, chunk)
            cqt = Resize(cqt, F_max, chunk)
            chroma = Resize(chroma, F_max, chunk)
            feat = torch.cat((mel_specs, cqt, chroma), dim=1)
        else:
            feat = (mel_specs, cqt, chroma)
    return feat, genres, genre_tops
