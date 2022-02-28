import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        # self.apply(weights_init)

    def init_hidden(self):
        # make hidden states on same device as model
        # the new() here api is not elegant
        # todo
        # return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # 通常input应该是四维数据 n_episode * 1_tansition * n_agent * n_observation
        # 可以是三维b代表batch_size, a 代表agent， e代表oberservation维度
        # 这里应该没有n_agent, 推测
        b, a, e = inputs.size()
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)  # (b*a, e) --> (b*a, h)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)   #(b*a, h)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q.view(b, a, -1), h.view(b, a, -1)