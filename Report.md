# Implementation description

## Learning algorithm

Multi DDPG agent algorithm with random sample replay buffer (shared/different) is used for the project[2][1].  

![DDPG algorithm](ddpg_algo.png)


For our implementation we have used the following architecture:

* Actor: 512 -> 256 (with dropout=0.2, while training)
* Critic: 512 -> 256 -> 128 (with dropout=0.2, while training)
* Ornstein-Uhlenbeck noise


### Hyperparameters

```python
Config.tau = 1e-3                     # for soft update of target parameters
Config.weight_decay = 0.              # critic L2 weight decay
Config.states = None                  # environment states
Config.state_size = None              # environment state size
Config.action_size = None             # environment action size
Config.lr_actor = 1e-4                # actor agent learning rate
Config.lr_critic = 3e-4               # critic agent learning rate
Config.batch_size = 512               # minibatch size
Config.buffer_size = int(1e6)         # replay buffer size
Config.gamma = 0.99                   # discount factor
Config.update_every = 4               # how often to update the network
```

## Plots for the rewards

The following plots summarises the learning process of the agent through different episodes. The agent took 50 episodes to solve the environment.
![Average score every 100 episodes](logs.png)
![Score vs Episode](plot.png)

## Ideas for future work

Following optimizations can be applied to the project:

- Test and compare Reinforce model
- Implement prioritized replay buffer
- Paramter space noise for better exploration
- Test shared network between agents
- Estimate hyperparameters by training an agent
- Experiment with Generalized Advantage Estimation
- Experiment with A2c and A3C
- Test shared network between agents
- Test separate replay buffer for agents


### Refrences
1. [Deep Deterministic Policy Gradients in TensorFlow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
2. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)