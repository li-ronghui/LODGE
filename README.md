<div align="center">
<h2><font color="red"> 🕺🕺🕺 Lodge 💃💃💃 </font></center> <br> <center>A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives (CVPR 2024)</h2>

[Ronghui Li](https://mayuelala.github.io/), [Yuxiang Zhang](https://zhangyux15.github.io/), [Yachao Zhang](https://yachao-zhang.github.io/), [Hongwen Zhang](https://zhanghongwen.cn/), [Jie Guo](https://scholar.google.com/citations?hl=en&user=9QLVTUYAAAAJ), [Yan Zhang](https://yz-cnsdqz.github.io/),  [Yebin Liu](https://www.liuyebin.com/) and [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=zh-CN)

<a href='https://li-ronghui.github.io/lodge'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://li-ronghui.github.io/lodge'><img src='https://img.shields.io/badge/ArXiv-2304.01186-red'></a> 
<!-- ![visitors](https://visitor-badge.laobi.icu/badge?page_id=li-ronghui.LODGE&left_color=green&right_color=red)   -->
<!-- [![GitHub](https://img.shields.io/github/stars/li-ronghui/LODGE?style=social)](https://github.com/li-ronghui/LODGE)  -->
</div>





## 💃💃💃 Abstract
<b>TL;DR: We propose a two-stage diffusion modle that can generate extremely long dance from given music in a parallel manner.</b>

<details><summary>CLICK for full abstract</summary>

> We propose Lodge, a network capable of generating extremely long dance sequences conditioned on given music. We design Lodge as a two-stage coarse to fine diffusion architecture, and propose the characteristic dance primitives that possess significant expressiveness as intermediate representations between two diffusion models. The first stage is global diffusion, which focuses on comprehending the coarse-level music-dance correlation and production characteristic dance primitives. In contrast, the second-stage is the local diffusion, which parallelly generates detailed motion sequences under the guidance of the dance primitives and choreographic rules. In addition, we propose a Foot Refine Block to optimize the contact between the feet and the ground, enhancing the physical realism of the motion. Our approach can parallelly generate dance sequences of extremely long length, striking a balance between global choreographic patterns and local motion quality and expressiveness. Extensive experiments validate the efficacy of our method.
</details>


## 🎤🎤🎤 Todo

- [X] Release the code and config for teaser
- [ ] Release the checkpoints
- [ ] Release detailed guidance for traing and testing
- [ ] Release more applications


## 🍻🍻🍻 Setup Environment
Our method is trained using cuda11, pytorch-lightning 1.9.5 on Nvidia A100.
``` 
conda env create -f lodge.yml
``` 
Our environment is similar to EDGE ([official])(https://edge-dance.github.io/). You may check them for more details.

## 🔢🔢🔢 Data preparation

The [FineDance](https://github.com/li-ronghui/FineDance) dataset lasts an average of 152.3 seconds per dance and has a wealth of 22 dance genres, making it ideal for training dance generation, especially long dance generation. Therefore, we mainly use FineDance to conduct experiments. Please visit [here](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view) to download the origin FineDance dataset and put it in the ./data floder. Please notice that the origin FineDance motion has 52 joints (including 22 body joints and 30 hand joints), we only use the body part dance to train and test Lodge. Therefore, you need to run the following script to preprocess the dataset.

```bash
python data/code/preprocess.py
python dld/data/pre/FineDance_normalizer.py
```

Otherwise, directly download our preprocessed music and dance features from [here](https://drive.google.com/drive/folders/1cdj8YymfN1BHgggVfGaLjaa9vaEpjPzZ?usp=sharing) and put them into the ./data/finedance folder if you don't wish to process the data.

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

```bash
python train.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml 
```

## 🕺🕺🕺 Inference
Once the training is done, run inference:
The --soft is a float parameter range from 0 to 1, which can set the number of steps for the soft cue guidance action. 
```bash
python infer_lodge.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml --soft 1.0
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
<!-- ```bibtex
@article{ma2023follow,
  title={Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos},
  author={Ma, Yue and He, Yingqing and Cun, Xiaodong and Wang, Xintao and Shan, Ying and Li, Xiu and Chen, Qifeng},
  journal={arXiv preprint arXiv:2304.01186},
  year={2023}
}
```  -->


## 👯👯👯 Acknowledgements

This basic dance diffusion borrows from [EDGE](https://github.com/Stanford-TML/EDGE), the evaluate code borrows from  [Bailando](https://github.com/lisiyao21/Bailando). the README.md style borrows from [follow-your-pose](https://github.com/follow-your-pose/follow-your-pose.github.io). Thanks the authors for sharing their code and models.
