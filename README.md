# World Models A3C

## Implementation of a variant of World Models

![](/assets/world-models.png)

## Note
- Replaced MDN-RNN to LSTM for Memory
- Replaced CMA-ES to A3C for Controller
- Trained over two stages
    - Stage 1: V and M were trained on dataset with random rollout
    - Stage 2: V and M were trained on dataset with a3c rollout


## Training Result

<b>Result with dataset using random rollout</b>

<!-- ![](/assets/scores.png) -->
<p><img src="/assets/scores.png" width="400"></p>

<b>Result with dataset using the pretrained model rollout</b>

<p><img src="/assets/scores-additional.png" width="400"></p>
<!-- ![](/assets/scores-additional.png) -->

<b>Play Demo</b>

<p><img src="/assets/a3c.gif" width="400"></p>
<!-- ![](/assets/a3c.gif) -->


---

## Environment Setting
    apt-get update
    apt-get install swig
    pip install gym[box2d]


## Training Stage I
### Dataset Generation using Rollout with random policy
    python rollout.py

### Vision model with VAE
    python train-vae.py

### Memory model with LSTM-RNN
    python train-rnn.py

### Controller with A3C
    python train-a3c.py


## Training Stage II
### Rollout with the pretrained model
    python rollout-a3c.py


### Fine-tuning V and M
    python train-vae.py
    python train-rnn.py

### Train C with the improved V and M
    python train-a3c.py

## Test
    # <# of plays> <seed> <is_record>
    python test.py 2 999 False