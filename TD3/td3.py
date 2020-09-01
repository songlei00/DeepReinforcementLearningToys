from tools.memory import Memory 
from tools.agent import BaseAgent
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F 


def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)

def MLP(self, n_inputs, n_outputs, n_hidden=256, active_layer=nn.ReLU, output_active_layer=None):
    net = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        active_layer(),
        nn.Linear(n_hidden, n_hidden),
        active_layer(),
        nn.Linear(n_hidden, n_outputs)
    )
    net.apply(weight_init)
    return net


class DeterministicPolicy(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden):
        super(DeterministicPolicy, self).__init__()
        self.net = MLP(n_inputs, n_outputs)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.tanh(self.net(x))
        return x


class ClippedDoubleQFunction(nn.Module):
    """Clipped double Q to deal with overestimation"""

    def __init__(self, n_inputs, n_outputs, n_hidden=256):
        super(QNetwork, self).__init__()
        self.q1_net = MLP(n_inputs, n_outputs, n_hidden)
        self.q2_net = MLP(n_inputs, n_outputs, n_hidden)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x1 = self.q1_net(x)
        x2 = self.q2_net(x)

        return x1, x2

    
class TD3(BaseAgent):

    def __init__(
        self,
        env_name,
        env,
        lr=3e-4,
        gamma=0.99,
        tau=0.005
    ):
        BaseAgent.__init__(self)
        self.env_name = env_name
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]

        self.actor = DeterministicPolicy(self.n_state, self.n_action)
        self.target_actor = DeterministicPolicy(self.n_state, self.n_action)
        self.update_target(self.target_actor, self.actor)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = ClippedDoubleQFunction(self.n_state+self.n_action, 1)
        self.target_critic = ClippedDoubleQFunction(self.n_state+self.n_action, 1)
        self.update_target(self.target_critic, self.critic)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = Tensor(state).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy()

    
