# Leapfrog Diffusion Model for Stochastic Trajectory Prediction （LED）

Official **PyTorch** code for CVPR'23 paper "Leapfrog Diffusion Model for Stochastic Trajectory Prediction".


## 1. Overview

<div align="center">  
  <img src="./results/fig/overview.png" width = "60%" alt="system overview"/>
</div>

**Abstract**: To model the indeterminacy of human behaviors, stochastic trajectory prediction requires a sophisticated multi-modal distribution of future trajectories. Emerging diffusion models have revealed their tremendous representation capacities in numerous generation tasks, showing potential for stochastic trajectory prediction. However, expensive time consumption prevents diffusion models from real-time prediction, since a large number of denoising steps are required to assure sufficient representation ability. To resolve the dilemma, we present **LEapfrog Diffusion model (LED)**, a novel diffusion-based trajectory prediction model, which provides  real-time, precise, and diverse predictions. The core of the proposed LED is to leverage a trainable leapfrog initializer to directly learn an expressive multi-modal distribution of future trajectories, which skips a large number of denoising steps, significantly accelerating inference speed. Moreover, the leapfrog initializer is trained to appropriately allocate correlated samples to provide a diversity of predicted future trajectories, significantly improving prediction performances. Extensive experiments on four real-world datasets, including NBA/NFL/SDD/ETH-UCY, show that LED consistently improves performance and achieves **23.7\%/21.9\%** ADE/FDE improvement on NFL. The proposed LED also speeds up the inference **19.3/30.8/24.3/25.1** times compared to the standard diffusion model on NBA/NFL/SDD/ETH-UCY, satisfying real-time inference needs.

<div  align="center">  
  <img src="./results/fig/mean_var_estimation.png" width = "50%" alt="mean and variance estimation"/>
</div>
Here, we present an example (above) to illustrate the mean and variance estimation in the leapfrog initializer under four scenes on the NBA dataset. We see that the variance estimation can well describe the scene complexity for the current agent by the learned variance, showing the rationality of our variance estimation.


## 2. Code Guidance

Overall project structure:
```text
----LED\   
    |----README.md
    |----requirements.txt # packages to install                    
    |----main_led_nba.py  # [CORE] main file
    |----trainer\ # [CORE] main training files, we define the denoising process HERE!
    |    |----train_led_trajectory_augment_input.py 
    |----models\  # [CORE] define models under this file
    |    |----model_led_initializer.py                    
    |    |----model_diffusion.py    
    |    |----layers.py
    |----utils\ 
    |    |----utils.py 
    |    |----config.py
    |----data\ # preprocessed data (~200MB) and dataloader
    |    |----files\
    |    |    |----nba_test.npy
    |    |    |----nba_train.npy
    |    |----dataloader_nba.py
    |----cfg\ # config files
    |    |----nba\
    |    |    |----led_augment.yml
    |----results\ # store the results and checkpoints (~100MB)
    |----visualization\ # some visualization codes
```

Please download the data and results from [Google Drive](https://drive.google.com/drive/folders/1Uy8-WvlCp7n3zJKiEX0uONlEcx2u3Nnx?usp=sharing). 

**TODO list**:

- [ ] add training/evaluation for diffusion models (in two weeks).
- [ ] more detailed descripition in trainers (in one month).
- [ ] transfer the parameters in models into yaml.
- [ ] other fast sampling methods (DDIM and PD).


### 2.1. Environment
We train and evaluate our model on `Ubuntu=18.04` with `RTX 3090-24G`.

Create a new python environment (`led`) using `conda`:
```
conda create -n led python=3.7
conda activate led
```

Install required packages using Command 1 or 2:
```bash
# Command 1 (recommend):
pip install -r requirements.txt

# Command 2:
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install easydict
pip install glob2
```



### 2.2. Training

You can use the following command to start training the initializer.
```bash
python main_led_nba.py --cfg <-config_file_name_here-> --gpu <-gpu_index_here-> --train 1 --info <-experiment_information_here->

# e.g.
python main_led_nba.py --cfg led_augment --gpu 0 --train 1 --info try1
```

And the results are stored under the `./results` folder.



### 2.3. Evaluation

We provide pretrained models under the `./checkpoints` folder.

**Reproduce**. Using the command `python main_led_nba.py --cfg led_augment --gpu 0 --train 0 --info reproduce` and you will get the following results:
```text
(191888, 30, 35, 4)
(191888, 30, 35, 4)
500
(47972, 30, 35, 4)
(47972, 30, 35, 4)
100
[Core Denoising Model] Trainable/Total: 6568720/6568720
[Initialization Model] Trainable/Total: 4634997/4634997
./results/led_augment/New_diff_SinD_train_100epoch_withoutlight_t2/models/model_0100.p
--ADE(1s): 0.0284	--FDE(1s): 0.0255
--ADE(2s): 0.0333	--FDE(2s): 0.0371
--ADE(3s): 0.0395	--FDE(3s): 0.0485
--ADE(4s): 0.0471	--FDE(4s): 0.0720
```

### 2.4. Datasets
We use SinD Dataset to retrain the LED model, then evaluate it.
So far, we have extracted the data of pedestrian, bicycle, tricycle and motorcycle from SinD and saved them as SinD_7_train_light_p.npy, SinD_5_train_light_b.npy, SinD_20_train_light_t.npy, SinD_3_train_light_m.npy.
You can load the .npy data in the dataloader_nba.py
The processed data format is aligned with NBA dataset, which contains 11 agents' continuous 30-frames-length trajectories in a period of time. For the SinD dataset, we have to add blank data into the .npy, because the number of the agents in one frame is fluctuating. The number of the agents will be the largest appeared one during the recorded peroid.

### 2.5. Modifications
#### 2.5.1 Traffic Light Signal
Add Acceleration and Traffic Light Signals to the input feature layers.
Code need to be modified during running:

trainer/train_led_trajectory_augment_input.py
models/model_led_initializer.py
models/layers.py
models/model_diffusion.py

find #changing the mode, with or without light and acceleration

#### 2.5.2 Attention Module
Add a Squeeze-and-Excitation Module to self enhance the important features.
Code need to be modified during running:

models/model_led_initializer.py
line 25: #turn on or off the resnet and se module

#### 2.5.3 SinD Dataset Visualization
Add a python program to import and visualize the sind data set to make it easy to observe.
Code need to be modified during running:

Create a new python environment (`SinD`) using `conda`:
```
conda create -n SinD python=3.8
conda activate SinD
```
Cd into /LED_Modified/SinD/SIND-Vis-tool/ then install required packages:
```
pip install -r requirements.txt
```
To visualize the data run 
```
python VisMain.py <data_path (../LED_Modified/data/SinD/)> <record_name (7_28_1)> from this folder directory to visualize the recorded data.
```

In the visualization, traffic participants (vehicles and pedestrians) are presented as rectangular boxes. By clicking a rectangular box with the mouse, multiple graphs showing the changes of the parameters corresponding to the motion state of the traffic participants can pop up. 

## 3. Citation
If you find this code useful for your research, please cite this paper:

```bibtex
@inproceedings{mao2023leapfrog,
  title={Leapfrog Diffusion Model for Stochastic Trajectory Prediction},
  author={Mao, Weibo and Xu, Chenxin and Zhu, Qi and Chen, Siheng and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5517--5526},
  year={2023}
}
```

## 4. Acknowledgement

Most code is borrowed from [SIND] (https://github.com/SOTIF-AVLab/SinD.git) , [LED] (https://github.com/MediaBrain-SJTU/LED), [MID](https://github.com/Gutianpei/MID), [NPSN](https://github.com/InhwanBae/NPSN) and [GroupNet](https://github.com/MediaBrain-SJTU/GroupNet). We thank the authors for releasing their code.


[![Star History Chart](https://api.star-history.com/svg?repos=MediaBrain-SJTU/LED&type=Date)](https://star-history.com/#MediaBrain-SJTU/LED&Date)
