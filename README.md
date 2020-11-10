# MA-Minigrid-MADDPG
An implementation of [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch) on a multi-agent minigrid environment.

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* [shariqiqbal2810's fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 1.6.0
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.6
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 2.3.0 and [tensorboardX](https://github.com/lanpa/tensorboard-pytorch), version: 2.1 (for logging)
* My [fork](https://github.com/TheNeeloy/ma-minigrid) of MiniGrid

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

To train an example in the MiniGrid env:
```
ma-minigrid-maddpg: python main.py --env_id MiniGrid-MA-UnlockDoorGoalA2-v0 --model_name test_minigrid --discrete_action
```
To visualize a model from the MiniGrid env:
```
python evaluate.py --env_id MiniGrid-MA-UnlockDoorGoalA2-v0 --model_name test_minigrid
```
To train an example in the MPE env:
```
python main.py --mpe
```
To visualize a model from the MiniGrid env:
```
python evaluate.py --mpe
```

## Acknowledgements

[maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch) & all acknowledgements from that repo as well.