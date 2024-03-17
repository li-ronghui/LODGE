import os
from functools import partial
from pathlib import Path

import librosa
import librosa as lr
import numpy as np
from tqdm import tqdm
import soundfile as sf


HOP_LENGTH = 512
# SR = FPS * HOP_LENGTH
EPS = 1e-6


def slice_audio(audio_file, stride, length, out_dir, fps=30):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=fps*HOP_LENGTH)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx

def slice_musicfea(audio, stride, length):
    audio_slice = []
    start_idx = int(0)
    print("len", len(audio))
    print("length", length)
    while start_idx <= len(audio) - length:
        audio_slice.append(audio[int(start_idx) : int(start_idx + length)])
        start_idx += stride
        # print("start_idx", start_idx)
    return audio_slice


def _get_tempo(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""

    # a lot of stuff, only take the 5th element
    audio_name = audio_name.split("_")[4]

    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    else:
        assert False, audio_name


def extract(fpath, skip_completed=True, dest_dir="music_baseline_feats", fps=30):
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return
    SR = fps*HOP_LENGTH
    data, _ = librosa.load(fpath, sr=SR)

    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
    ).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    try:
        start_bpm = _get_tempo(audio_name)
    except:
        # determine manually
        start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )

    # chop to ensure exact shape
    # audio_feature = audio_feature[:5 * FPS]
    # assert (audio_feature.shape[0] - 5 * FPS) == 0, f"expected output to be ~5s, but was {audio_feature.shape[0] / FPS}"
    # os.makedirs(dest_dir, exist_ok=True)
    # np.save(save_path, audio_feature)
    return audio_feature, save_path