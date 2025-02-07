import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FCNetwork(nn.Module):
    def __init__(self, inp_dim, hidden_dims, out_dim, act_fn=nn.ReLU()):
        super(FCNetwork, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.learn = True

        layer_lst = []
        in_dim = inp_dim

        for hidden_dim in hidden_dims:
            layer_lst.append(nn.Linear(in_dim, hidden_dim))
            layer_lst.append(act_fn)
            in_dim = hidden_dim

        # layer_lst.append(layer_init(nn.Linear(in_dim, out_dim), std=0.01))
        layer_lst.append(nn.Linear(in_dim, out_dim))

        self.network = nn.Sequential(*layer_lst)

    def forward(self, inp):
        return self.network(inp)

    @property
    def num_layers(self):
        return len(self.hidden_dims) + 1


class PolicyNetwork(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dims, act_dim, masked_dim=0, act_fn=nn.ReLU()):
        super(PolicyNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.masked_dim = masked_dim
        self.base = FCNetwork(inp_dim=latent_dim + state_dim - masked_dim, hidden_dims=hidden_dims, out_dim=act_dim,
                              # 2 * act_dim,
                              act_fn=act_fn)
        self.logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, latent, state):
        if state is None:
            inp = latent
        elif latent is None:
            inp = state[:, self.masked_dim:]
        else:
            inp = torch.cat([latent, state[:, self.masked_dim:]], dim=1)
        # base_out = self.base(inp)
        # mean = base_out[:, :self.act_dim]
        mean = self.base(inp)
        # log_std = base_out[:, self.act_dim:]
        log_std = self.logstd.expand_as(mean)
        std = torch.exp(log_std)
        # std = log_std.exp()
        return mean, std

    def act(self, latent, state, deterministic=False):
        mean, std = self.forward(latent, state)
        if deterministic:
            return mean.detach()
        else:
            act_dist = torch.distributions.Normal(mean, std)
            latent = act_dist.sample()
            return latent, act_dist

    def calc_log_prob(self, latent, state, action):
        mean, std = self.forward(latent, state)
        act_dist = torch.distributions.Normal(mean, std)
        log_prob = act_dist.log_prob(action).sum(-1)
        return log_prob.mean()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ConvEncoder(nn.Module):
    def __init__(self, inp_dim, out_dim=256):
        super(ConvEncoder, self).__init__()
        self.state_dim = inp_dim
        self.out_dim = out_dim
        c, h, w = self.state_dim
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=self.out_dim),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, inp):
        return self.network(inp)


class CnnPolicyNetwork(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dims, act_dim, flat_state_dim=256, act_fn=nn.ReLU()):
        super(CnnPolicyNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.flat_state_dim = flat_state_dim

        self.state_encoder = ConvEncoder(inp_dim=state_dim, out_dim=flat_state_dim)
        self.base = FCNetwork(inp_dim=latent_dim + flat_state_dim, hidden_dims=hidden_dims, out_dim=act_dim,
                              act_fn=act_fn)
        self.logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, latent, state, encode_state=False):
        if state is None:
            inp = latent
        else:
            if encode_state:
                state = state.permute(
                    (0, 1, 2, 3)) / 255.0  # "batch traj height width channel" -> "batch traj channel height width"
                state = self.state_encoder(state)
            if latent is None:
                # print("prior")
                # print(state)
                inp = state
            else:
                inp = torch.cat([latent, state], dim=1)

        mean = self.base(inp)
        log_std = self.logstd.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

    def act(self, latent, state, encode_state=False):
        if latent is not None:
            # act for the decoder net
            logits, _ = self.forward(latent, state, encode_state)
            act_dist = Categorical(logits=logits)
            sample = act_dist.sample()
        else:
            # act for the prior net
            mean, std = self.forward(latent, state, encode_state)
            act_dist = torch.distributions.Normal(mean, std)
            sample = act_dist.sample()
        return sample, act_dist

    def calc_log_prob(self, latent, state, action, encode_state=False):
        logits, _ = self.forward(latent, state, encode_state)
        act_dist = Categorical(logits=logits)
        log_prob = act_dist.log_prob(action)#.sum(-1)
        return log_prob.mean()


class ARPolicyNetwork(nn.Module):
    # Doesn't have masked dim
    def __init__(self, latent_dim, state_dim, act_dim, low_act=-1.0, up_act=1.0, act_fn=nn.ReLU()):
        super(ARPolicyNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.state_embed_hid = 256
        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, self.state_embed_hid),
            act_fn,
            nn.Linear(self.state_embed_hid, self.state_embed_hid),
            act_fn,
            nn.Linear(self.state_embed_hid, self.state_embed_hid),
            act_fn,
            nn.Linear(self.state_embed_hid, self.state_embed_hid),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.act_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(self.state_embed_hid, self.action_embed_hid), act_fn,
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.act_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(self.state_embed_hid + self.out_lin, self.action_embed_hid), act_fn,
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, latent, state, deterministic=False):
        if state is None:
            state_inp = latent
        elif latent is None:
            state_inp = state
        else:
            state_inp = torch.cat([latent, state], dim=1)

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.act_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def act(self, latent, state, deterministic=False):
        return self.forward(latent, state, deterministic)

    def calc_log_prob(self, latent, state, action):
        l_action = ((action - self.low_act) // self.bin_size).long()

        if state is None:
            state_inp = latent
        elif latent is None:
            state_inp = state
        else:
            state_inp = torch.cat([latent, state], dim=1)

        state_d = self.state_embed(state_inp)
        log_prob = -self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])
        for i in range(1, self.act_dim):
            log_prob -= self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return log_prob


class LMP(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, hidden_dims, tanh=False, latent_reg=0.0, ar=False,
                 ar_params=None, rnn_layers=4, goal_idxs=None, act_fn=nn.ReLU()):
        super(LMP, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        self.tanh = tanh
        self.latent_reg = latent_reg
        self.act_fn = act_fn
        self.ar = ar
        self.create_encoder()
        self.decoder = PolicyNetwork(latent_dim=latent_dim, state_dim=self.state_dim, hidden_dims=hidden_dims,
                                         act_dim=action_dim, act_fn=act_fn)
        self.prior = PolicyNetwork(latent_dim=0, state_dim=self.state_dim, hidden_dims=hidden_dims, act_dim=latent_dim,
                                   act_fn=act_fn)

    def create_encoder(self):
        self.state_encoder = FCNetwork(inp_dim=self.state_dim, hidden_dims=self.hidden_dims,
                                       out_dim=self.hidden_dims[-1], act_fn=self.act_fn)
        input_dim = self.hidden_dims[-1] + self.action_dim
        self.birnn_encoder = nn.GRU(input_dim, self.hidden_dims[-1], self.rnn_layers, batch_first=True,
                                    bidirectional=True)

        self.mean_encoder = nn.Linear(2 * self.hidden_dims[-1], self.latent_dim)
        self.logstd_encoder = nn.Linear(2 * self.hidden_dims[-1], self.latent_dim)

    def forward_encoder(self, state_traj, action_traj):
        if state_traj is None:
            inp = action_traj
        else:
            batch_size, seq_len = state_traj.size(0), state_traj.size(1)
            state_traj = state_traj.view(-1, self.state_dim)
            state_traj = self.state_encoder(state_traj)
            state_traj = state_traj.view(batch_size, seq_len, -1)
            inp = torch.cat([state_traj, action_traj], dim=2)

        out_birnn, _ = self.birnn_encoder(inp)
        h = torch.cat([out_birnn[:, -1, :self.hidden_dims[-1]], out_birnn[:, 0, self.hidden_dims[-1]:]], dim=1)
        mean = self.mean_encoder(h)
        log_std = self.logstd_encoder(h)
        return mean, log_std.exp()

    def calc_loss(self, state_traj, action_traj, is_cuda):
        # Assumes all traj is of equal length
        batch_size, seq_len = action_traj.size(0), action_traj.size(1)
        nll_loss, kl_loss = 0., 0.
        encoder_mean, encoder_std = self.forward_encoder(action_traj=action_traj, state_traj=state_traj)
        prior_mean, prior_std = self.prior(latent=None, state=state_traj[:, 0, :])
        encoder_p = torch.distributions.Normal(encoder_mean, encoder_std)
        prior_p = torch.distributions.Normal(prior_mean, prior_std)

        kl_loss = torch.distributions.kl_divergence(encoder_p, prior_p).sum()
        kl_loss = torch.clamp(kl_loss, -100, 100)

        zeros = torch.zeros(batch_size, self.latent_dim)
        ones = torch.ones(batch_size, self.latent_dim)
        if is_cuda:
            zeros = zeros.cuda()
            ones = ones.cuda()
        eps_dist = torch.distributions.Normal(zeros, ones)
        latent = encoder_mean + eps_dist.sample() * encoder_std

        if self.tanh:
            latent = torch.tanh(latent)

        for t in range(seq_len):
            nll_loss -= self.decoder.calc_log_prob(latent=latent, state=state_traj[:, t], action=action_traj[:, t])

        nll_loss = nll_loss / seq_len
        nll_loss = torch.clamp(nll_loss, -100, 100)

        if self.latent_reg > 0:
            nll_loss += self.latent_reg * latent.norm(p=2, dim=1).mean()

        return kl_loss, nll_loss


class CnnLMP(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, hidden_dims, flat_state_dim=256, tanh=False, latent_reg=0.0,
                 ar=False, rnn_layers=4, act_fn=nn.ReLU()):
        super(CnnLMP, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.flat_state_dim = flat_state_dim
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        self.tanh = tanh
        self.latent_reg = latent_reg
        self.act_fn = act_fn
        self.ar = ar
        self.create_encoder()
        self.decoder = CnnPolicyNetwork(latent_dim=latent_dim, state_dim=self.state_dim, hidden_dims=hidden_dims,
                                            act_dim=action_dim, act_fn=act_fn)
        self.prior = CnnPolicyNetwork(latent_dim=0, state_dim=self.state_dim, hidden_dims=hidden_dims,
                                      act_dim=latent_dim, act_fn=act_fn)

    def create_encoder(self):
        self.state_encoder = ConvEncoder(inp_dim=self.state_dim, out_dim=self.flat_state_dim)

        input_dim = self.flat_state_dim + 1  # self.action_dim
        self.birnn_encoder = nn.GRU(input_dim, self.hidden_dims[-1], self.rnn_layers, batch_first=True,
                                    bidirectional=True)

        self.mean_encoder = nn.Linear(2 * self.hidden_dims[-1], self.latent_dim)
        self.logstd_encoder = nn.Linear(2 * self.hidden_dims[-1], self.latent_dim)

    def forward_encoder(self, state_traj, action_traj):
        if state_traj is None:
            inp = action_traj
        else:
            state_traj = state_traj.permute(
                (0, 1, 2, 3, 4)) / 255.0  # "batch traj height width channel" -> "batch traj channel height width"
            batch_size, seq_len = state_traj.size(0), state_traj.size(1)
            state_traj = state_traj.view(-1, state_traj.size(2), state_traj.size(3), state_traj.size(4))
            state_traj = self.state_encoder(state_traj)
            state_traj = state_traj.view(batch_size, seq_len, -1)
            action_traj = action_traj.view(batch_size, seq_len, -1) / 14.0
            inp = torch.cat([state_traj, action_traj], dim=2)

        out_birnn, _ = self.birnn_encoder(inp)
        h = torch.cat([out_birnn[:, -1, :self.hidden_dims[-1]], out_birnn[:, 0, self.hidden_dims[-1]:]], dim=1)
        mean = self.mean_encoder(h)
        log_std = self.logstd_encoder(h)
        return mean, log_std.exp()

    def calc_loss(self, state_traj, action_traj, is_cuda):
        # Assumes all traj is of equal length
        batch_size, seq_len = action_traj.size(0), action_traj.size(1)
        nll_loss, kl_loss = 0., 0.
        encoder_mean, encoder_std = self.forward_encoder(action_traj=action_traj, state_traj=state_traj)
        prior_mean, prior_std = self.prior(latent=None, state=state_traj[:, 0, :], encode_state=True)
        encoder_p = torch.distributions.Normal(encoder_mean, encoder_std)
        prior_p = torch.distributions.Normal(prior_mean, prior_std)

        kl_loss = torch.distributions.kl_divergence(encoder_p, prior_p).sum()
        kl_loss = torch.clamp(kl_loss, -100, 100)

        zeros = torch.zeros(batch_size, self.latent_dim)
        ones = torch.ones(batch_size, self.latent_dim)
        if is_cuda:
            zeros = zeros.cuda()
            ones = ones.cuda()
        eps_dist = torch.distributions.Normal(zeros, ones)
        latent = encoder_mean + eps_dist.sample() * encoder_std

        if self.tanh:
            latent = torch.tanh(latent)

        for t in range(seq_len):
            nll_loss -= self.decoder.calc_log_prob(latent=latent, state=state_traj[:, t], action=action_traj[:, t],
                                                       encode_state=True)

        nll_loss = nll_loss / seq_len
        nll_loss = torch.clamp(nll_loss, -100, 100)

        if self.latent_reg > 0:
            nll_loss += self.latent_reg * latent.norm(p=2, dim=1).mean()

        return kl_loss, nll_loss
