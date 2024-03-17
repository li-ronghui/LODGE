'''
Extract music feature (35 dimension) for wav files.
'''
import librosa
import numpy as np
import os
import wave
from tqdm import  tqdm
import librosa as lr

FPS = 30
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6


audio_dir = 'data/finedance/music_wav'
target_dir_ori = "data/finedance/music_npy_test"
os.makedirs(target_dir_ori, exist_ok=True)


# AIST++
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

for file in tqdm(os.listdir(audio_dir)):
    audio_name = file[:-4] 

    save_path = os.path.join(target_dir_ori, f"{audio_name}.npy") ##存特征路径
    music_file = os.path.join(audio_dir, file)
    
    
    data, _ = librosa.load(music_file, sr=SR)

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
        start_bpm = lr.beat.tempo(y=lr.load(music_file)[0])[0]

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
    np.save(save_path, audio_feature)
    
    