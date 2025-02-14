<div align="center">
<h2><font color="red"> 🕺🕺🕺 Lodge 💃💃💃 </font></center> <br> <center>A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives (CVPR 2024)</h2>

[Ronghui Li](https://li-ronghui.github.io/), [Yuxiang Zhang](https://zhangyux15.github.io/), [Yachao Zhang](https://yachao-zhang.github.io/), [Hongwen Zhang](https://zhanghongwen.cn/), [Jie Guo](https://scholar.google.com/citations?hl=en&user=9QLVTUYAAAAJ), [Yan Zhang](https://yz-cnsdqz.github.io/),  [Yebin Liu](https://www.liuyebin.com/) and [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=zh-CN)

<a href='https://li-ronghui.github.io/lodge'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://arxiv.org/abs/2403.10518'><img src='https://img.shields.io/badge/ArXiv-2304.01186-red'></a> 
</div>





## 💃💃💃 Abstract
<b>TL;DR: We propose a two-stage diffusion model that can generate extremely long dance from given music in a parallel manner.</b>

<details><summary>CLICK for full abstract</summary>

> We propose Lodge, a network capable of generating extremely long dance sequences conditioned on given music. We design Lodge as a two-stage coarse to fine diffusion architecture, and propose the characteristic dance primitives that possess significant expressiveness as intermediate representations between two diffusion models. The first stage is global diffusion, which focuses on comprehending the coarse-level music-dance correlation and production characteristic dance primitives. In contrast, the second-stage is the local diffusion, which parallelly generates detailed motion sequences under the guidance of the dance primitives and choreographic rules. In addition, we propose a Foot Refine Block to optimize the contact between the feet and the ground, enhancing the physical realism of the motion. Our approach can parallelly generate dance sequences of extremely long length, striking a balance between global choreographic patterns and local motion quality and expressiveness. Extensive experiments validate the efficacy of our method.
</details>


## 🎤🎤🎤 Todo

- [X] Release the code and config for teaser
- [X] Release the checkpoints
- [X] Release detailed guidance for traing and testing
- [ ] Release more applications


## 🍻🍻🍻 Setup Environment
Our method is trained using cuda11, pytorch-lightning 1.9.5 on Nvidia A100.
``` 
conda env create -f lodge.yml
``` 
Our environment is similar to EDGE [official](https://edge-dance.github.io/). You may check them for more details.

## 🔢🔢🔢 Data preparation

The [FineDance](https://github.com/li-ronghui/FineDance) dataset lasts an average of 152.3 seconds per dance and has a wealth of 22 dance genres, making it ideal for training dance generation, especially long dance generation. Therefore, we mainly use FineDance to conduct experiments. Please visit [Google Driver](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view) or [百度云](https://pan.baidu.com/s/1gynUC7pMdpsE31wAwq177w?pwd=o9pw) to download the origin FineDance dataset and put it in the ./data floder. Please notice that the origin FineDance motion has 52 joints (including 22 body joints and 30 hand joints), we only use the body part dance to train and test Lodge. Therefore, you need to run the following script to preprocess the dataset.

```bash
python data/code/preprocess.py
python dld/data/pre/FineDance_normalizer.py
```

Otherwise, directly download our preprocessed music and dance features from [Google Driver](https://drive.google.com/drive/folders/1cdj8YymfN1BHgggVfGaLjaa9vaEpjPzZ?usp=sharing) or [百度云](https://pan.baidu.com/s/1PQ53ooKxp-EkQvhiv7SKcA?pwd=y0ly) and put them into the ./data/finedance folder if you don't wish to process the data.

The final file structure is as follows:

```bash
LODGE
├── data
│   ├── code
│   │   ├──preprocess.py
│   │   ├──extract_musicfea35.py
│   ├── finedance
│   │   ├──label_json
│   │   ├──motion
│   │   ├──music_npy
│   │   ├──music_wav
│   │   ├──music_npynew
│   │   ├──mofea319
│   │── Normalizer.pth
└   └── smplx_neu_J_1.npy
```





## 💃💃💃 Training

Traing the Local Diffusion and Global Diffusion
```bash
python train.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml 
python train.py --cfg configs/lodge/coarse_finedance_fea139.yaml --cfg_assets configs/data/assets.yaml
```

Set the pretrained Local Diffusion checkpoint path at the "TRAIN.PRETRAINED" of "configs/lodge/finedance_fea139_finetune_v2.yaml", then finetuning the Local Diffusion for smooth generation.
```bash
python train.py --cfg configs/lodge/finedance_fea139_finetune_v2.yaml --cfg_assets configs/data/assets.yaml
```

You can also download the pretrained model from [Google Driver](https://drive.google.com/file/d/13Yp__EPAw0EjrSS898X5FtSQGmveBykA/view?usp=sharing) or [百度云](https://pan.baidu.com/s/1twYAdqR5OjSPkIlT1AJafw?pwd=1mte).

## 🕺🕺🕺 Inference
Once the training is done, run inference:
The --soft is a float parameter range from 0 to 1, which can set the number of steps for the soft cue guidance action. 
```bash
python infer_lodge.py --cfg exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml --cfg_assets configs/data/assets.yaml --soft 1.0
```

## 🖥️🖥️🖥️ Rendering
```bash
python render.py --modir 'your motion dir'
```

## 🔎🔎🔎 Evaluate
Once the inference is done, run evaluate:

```bash
python metric/metrics_finedance.py
python metric/beat_align_score.py
python metric/foot_skating.py
```




## 🎼🎼🎼 Citation 
If you think this project is helpful, please leave a star⭐️⭐️⭐️ and cite our paper:
```bibtex
@inproceedings{li2024lodge,
  title={Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives},
  author={Li, Ronghui and Zhang, YuXiang and Zhang, Yachao and Zhang, Hongwen and Guo, Jie and Zhang, Yan and Liu, Yebin and Li, Xiu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1524--1534},
  year={2024}
}
@inproceedings{li2023finedance,
  title={Finedance: A fine-grained choreography dataset for 3d full body dance generation},
  author={Li, Ronghui and Zhao, Junfan and Zhang, Yachao and Su, Mingyang and Ren, Zeping and Zhang, Han and Tang, Yansong and Li, Xiu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10234--10243},
  year={2023}
}
``` 


## 👯👯👯 Acknowledgements

This basic dance diffusion borrows from [EDGE](https://github.com/Stanford-TML/EDGE), the evaluate code borrows from  [Bailando](https://github.com/lisiyao21/Bailando), the README.md style borrows from [follow-your-pose](https://github.com/mayuelala/FollowYourPose). Thanks the authors for sharing their code and models.
