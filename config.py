class Config:
  def __init__(self):
    self.device = 'cpu'
    self.seed = 1
    self.random_seed = 1
    self.num_agents = 1

    self.tau = 1e-3
    # critic weight decay
    self.weight_decay = 0.
    self.states = None
    self.state_size = None
    self.action_size = None
    self.lr_actor = 1e-4
    self.lr_critic = 3e-4
    self.batch_size = 512
    # memory buffer size
    self.buffer_size = int(1e6)
    self.gamma = 0.99
    self.update_every = 4
