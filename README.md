**Noise Distribution Decomposition**
![logo](logo.png)
Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

```
@article{geng2023noise,
title={Noise Distribution Decomposition based Multi-Agent Distributional Reinforcement Learning},
author={Geng, Wei and Xiao, Baidi and Li, Rongpeng and Wei, Ning and Wang, Dong and Zhao, Zhifeng},
journal={arXiv preprint arXiv:2312.07025},
year={2023}
}
```

**Features**

<ul><li>A framework about distribution decomposition based on MARL methods.</li><li>Implemented in the environment: Multi-agent Particle world Environments and StarCraft Multi-Agent Challenge.</li><li>Contains mainstream comparison algorithms of MARL (MAPPO, MADDPG, MATD3, QMIX, VDN).</li>
</ul>



**Requirements**

```
pip install -r requirements.txt
```



**Training**

```
python model_runner.py --env=mpe
```

