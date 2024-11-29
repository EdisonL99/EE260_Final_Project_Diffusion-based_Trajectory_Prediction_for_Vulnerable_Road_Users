# EE260 Final Project: Diffusion-based Trajectory Prediction for Vulnerable Road Users

Official **PyTorch** code for CVPR'23 paper "Leapfrog Diffusion Model for Stochastic Trajectory Prediction".

## 1. Overview

<div align="center">  
  <img src="./results/fig/overview.png" width = "60%" alt="system overview"/>
</div>

**Abstract**: With the development of the diffusion model, more and more interesting scenarios are proposed, e.g., image generation. Also, a few researchers consider generating trajectories using the diffusion model. In the meantime, with the rising attention on the safety issue of VRU, VRU trajectory prediction at the intersection should be further considered. Thus, this final project tries to use the diffusion-based model to predict trajectories of VRU at the intersection scenario, explore the affection of the traffic signals on the performance of the model, and compare the difference of the model’s performance in different VRU objects. As a result, the diffusion-based model can generate satisfactory trajectories, and the traffic signal affects the fast-speed object more.

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

```
models/model_led_initializer.py
models/layers.py
models/model_diffusion.py
```

Find #changing the mode, with or without light and acceleration

#### 2.5.2 Attention Module
Add a Squeeze-and-Excitation Module to self enhance the important features.
Code need to be modified during running:

```
models/model_led_initializer.py
```

Line 25: #turn on or off the ResNet and SE module
