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

## 💃💃💃 Training

```bash
python train.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml 
```

## 🕺🕺🕺 Inference
Once the training is done, run inference:

```bash
python infer_lodge.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml 
```






## 🎼🎼🎼 Citation 
If you think this project is helpful, please feel free to leave a star⭐️⭐️⭐️ and cite our paper:
<!-- ```bibtex
@article{ma2023follow,
  title={Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos},
  author={Ma, Yue and He, Yingqing and Cun, Xiaodong and Wang, Xintao and Shan, Ying and Li, Xiu and Chen, Qifeng},
  journal={arXiv preprint arXiv:2304.01186},
  year={2023}
}
```  -->


## 👯👯👯 Acknowledgements

This repository borrows heavily from [EDGE](https://github.com/Stanford-TML/EDGE) and [Bailando](https://github.com/lisiyao21/Bailando). Thanks the authors for sharing their code and models.
