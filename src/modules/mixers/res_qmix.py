import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResQMixer(nn.Module):
    def __init__(self, args):
        super(ResQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True)
        # self.abs = False
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, self.n_agents))
        

    def forward(self, agent_qs, states):
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        bs = agent_qs.size(0)
        agent_qs_bak = agent_qs     # 这里备份一个agent_qs 最初的形状以作残差连接
        states = states.reshape(-1, self.state_dim)
        # states shape (e*t, s)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # agent_qs(b*t, 1, self.n_agents) w1(b*t,self.n_agents, self.embed_dim),  b1(b*t, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, self.n_agents)
        # State-dependent bias
        v = self.V(states).view(-1, 1, self.n_agents)
        # Compute final output
        # hidden(b*t, 1, self.embed_dim) w_final(b*t,self.embed_dim, 1 else n_agent for q_mix ),  V(b*t, 1, self.n_agents)
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, self.n_agents)
        # 残差连接后返回的q_tot维度依然不变（b, t, n_agents）
        # todo 消融注释了一半
        return q_tot + agent_qs_bak
