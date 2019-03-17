import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic_MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Basic_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class RNN_tanh(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_tanh, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.o_mat = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.squeeze(0)
        hidden = torch.zeros(1, self.hidden_size)
        out = torch.Tensor([])
        if self.cuda_on:
            hidden = hidden.cuda()
            out = out.cuda()
        for i_step in range(x.size(0)):
            hidden = self.rnn(x[i_step, :].unsqueeze(0), hidden)
            out = torch.cat([out, hidden], 0)
        out = torch.sigmoid(self.o_mat(out))
        out = out.unsqueeze(0)
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.o_mat = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.squeeze(0)
        hidden = torch.zeros(1, self.hidden_size)
        cx = torch.zeros(1, self.hidden_size)
        out = torch.Tensor([])
        if self.cuda_on:
            hidden = hidden.cuda()
            cx = cx.cuda()
            out = out.cuda()
        for i_step in range(x.size(0)):
            hidden, cx = self.rnn(x[i_step, :].unsqueeze(0), (hidden, cx))
            out = torch.cat([out, hidden], 0)
        out = torch.sigmoid(self.o_mat(out))
        out = out.unsqueeze(0)
        return out


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.rnn = nn.GRUCell(input_size, hidden_size)
        self.o_mat = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.squeeze(0)
        hidden = torch.zeros(1, self.hidden_size)
        out = torch.Tensor([])
        if self.cuda_on:
            hidden = hidden.cuda()
            out = out.cuda()
        for i_step in range(x.size(0)):
            hidden = self.rnn(x[i_step, :].unsqueeze(0), hidden)
            out = torch.cat([out, hidden], 0)
        out = torch.sigmoid(self.o_mat(out))
        out = out.unsqueeze(0)
        return out


class RNN_tanh2(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_tanh2, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.u_mat = nn.Linear(input_size, hidden_size)
        nn.init.normal_(self.u_mat.weight, mean=normal_mean, std=normal_std)
        self.w_mat = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.w_mat.weight, mean=normal_mean, std=normal_std)
        self.o_mat = nn.Linear(hidden_size, input_size)
        nn.init.normal_(self.o_mat.weight, mean=normal_mean, std=normal_std)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.hidden_size)
        if self.cuda_on:
            hidden = hidden.cuda()
        out = None
        for i_step in range(x.size(1)):
            a = torch.add(self.u_mat(x[:, i_step, :]), self.w_mat(hidden))
            hidden = torch.tanh(a)
            if out is None:
                out = hidden.unsqueeze(1)
            else:
                out = torch.cat([out, hidden.unsqueeze(1)], 1)

        out = torch.sigmoid(self.o_mat(out))
        return out


class RNN_LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.uf_mat = nn.Linear(input_size, hidden_size)
        self.wf_mat = nn.Linear(hidden_size, hidden_size)
        self.ug_mat = nn.Linear(input_size, hidden_size)
        self.wg_mat = nn.Linear(hidden_size, hidden_size)
        self.u_mat = nn.Linear(input_size, hidden_size)
        self.w_mat = nn.Linear(hidden_size, hidden_size)
        self.uo_mat = nn.Linear(input_size, hidden_size)
        self.wo_mat = nn.Linear(hidden_size, hidden_size)
        self.o_mat = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.hidden_size)
        stat = torch.zeros(x.size(0), self.hidden_size)
        if self.cuda_on:
            hidden = hidden.cuda()
            stat = stat.cuda()
        out = None
        for i_step in range(x.size(1)):
            forget_gate = torch.sigmoid(torch.add(self.uf_mat(x[:, i_step, :]), self.wf_mat(hidden)))
            external_input_gate = torch.sigmoid(torch.add(self.ug_mat(x[:, i_step, :]), self.wg_mat(hidden)))
            current_val = torch.tanh(torch.add(self.u_mat(x[:, i_step, :]), self.w_mat(hidden)))
            stat = forget_gate * stat + external_input_gate * current_val
            output_gate = torch.sigmoid(torch.add(self.uo_mat(x[:, i_step, :]), self.wo_mat(hidden)))
            hidden = torch.tanh(stat) * output_gate
            if out is None:
                out = hidden.unsqueeze(1)
            else:
                out = torch.cat([out, hidden.unsqueeze(1)], 1)

        out = torch.sigmoid(self.o_mat(out))
        return out


class RNN_GRU2(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_on=False, normal_mean=0.0, normal_std=0.075):
        super(RNN_GRU2, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_on = cuda_on
        self.uz_mat = nn.Linear(input_size, hidden_size)
        self.wz_mat = nn.Linear(hidden_size, hidden_size)
        self.u_mat = nn.Linear(input_size, hidden_size)
        self.w_mat = nn.Linear(hidden_size, hidden_size)
        self.ur_mat = nn.Linear(input_size, hidden_size)
        self.wr_mat = nn.Linear(hidden_size, hidden_size)
        self.o_mat = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.hidden_size)
        if self.cuda_on:
            hidden = hidden.cuda()
        out = None
        for i_step in range(x.size(1)):
            update_gate = torch.sigmoid(self.uz_mat(x[:, i_step, :]) + self.wz_mat(hidden))
            reset_gate = torch.sigmoid(self.ur_mat(x[:, i_step, :]) + self.wr_mat(hidden))
            current_val = torch.tanh(self.u_mat(x[:, i_step, :]) + self.w_mat(hidden * reset_gate))
            hidden = update_gate * hidden + (1 - update_gate) * current_val
            if out is None:
                out = hidden.unsqueeze(1)
            else:
                out = torch.cat([out, hidden.unsqueeze(1)], 1)

        out = torch.sigmoid(self.o_mat(out))
        return out
