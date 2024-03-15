import logging
import os
import sys
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
from functools import cmp_to_key
from tempfile import TemporaryDirectory

import numpy as np
import torch, glob, random
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from dld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from dld.data.get_data import get_datasets
# from dld.data.sampling import subsample, upsample
from dld.models.get_model import get_module
from dld.utils.logger import create_logger
from dld.data.utils.audio import slice_audio
from dld.data.utils.audio import extract as extract_music35



key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

def main():
    """
    get input music
    """
    # parse options
    cfg = parse_args(phase="demo")
    print("cfg.device", cfg.DEVICE)
    cfg.DEVICE = cfg.DEVICE[0]
    device = f"cuda:{cfg.DEVICE}"
    # sys.exit(0)
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")
    musicdir = cfg.DEMO.MusicDir
    # test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    test_list = ["063"]
    feature_func = extract_music35
    sample_length = cfg.FINEDANCE.full_seq_len * 8 /30      # 总长度，8 是指生成8段full_seq_len
    sample_size = int(sample_length / (cfg.FINEDANCE.full_seq_len/60) ) - 1
    print("sample_length", sample_length)
    print("sample_size", sample_size)

    temp_dir_list = []
    all_cond = []
    all_filenames = []

    if cfg.DEMO.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(cfg.DEMO.use_cached_features, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(musicdir, "*.wav")):
            songname = os.path.splitext(os.path.basename(wav_file))[0]
            if songname not in test_list:
                continue
            # create temp folder (or use the cache folder if specified)
            if cfg.DEMO.use_cached_features:
                
                save_dir = os.path.join(cfg.DEMO.use_cached_features, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, cfg.FINEDANCE.full_seq_len/60, cfg.FINEDANCE.full_seq_len/30, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - int(sample_size))
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not cfg.DEMO.use_cached_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)[:cfg.FINEDANCE.full_seq_len]
                reps = reps[:cfg.FINEDANCE.full_seq_len]
                print("reps", reps.shape)
                # save reps
                if cfg.DEMO.use_cached_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])



    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_long_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    # if cfg.ACCELERATOR == "gpu":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #         str(x) for x in cfg.DEVICE)
    #     device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_module(cfg, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.to(device)
    model.eval()


    start_time = time.time()
    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample_ori(
            data_tuple, "test", output_dir, render_count=-1, fk_out=output_dir, render=cfg.DEMO.RENDER, setmode="long"
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()
 

    total = time.time() - start_time
    print("total time is ", total)


if __name__ == "__main__":
    main()