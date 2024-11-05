import pennylane as qml
from math import ceil, pi
import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)

n_qubits = 4
n_layers = 2
dev = qml.device('default.qubit', wires=n_qubits)

def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)
        # qml.RY(inputs[qub], wires=qub)

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[layer,i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits,2*n_qubits):
            qml.RY(weights[layer,j], wires=j % n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class Quanv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers,2*n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method="best")
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        #self.stride = stride

    def forward(self, x):
        assert len(x.shape) == 4  # (bs, c, w, h)
        assert x.shape[2] >= self.kernel_size and x.shape[3] >= self.kernel_size

        bs = x.shape[0]
        c = x.shape[1]

        # side_len = X.shape[2] - self.kernel_size + 1  # *******
        x_lst = []
        for i in range(0, x.shape[2]-1,2):
            for j in range(0, x.shape[3]-1,2):
                x_lst.append(self.ql1(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1)))
        x = torch.cat(x_lst,dim=1)  # .view(bs,n_qubits,14,14)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.qc = Quanv2d(kernel_size=2)  # 假设Quanv2d类已经正确定义并可以工作

        # 增加全连接层的数量和每层的神经元数量
        # 假设经过self.qc后，输出的特征维度是n_qubits * (经过处理的图像块数量)
        # 这里我们假设经过处理的图像块数量为6（根据Quanv2d类的实现，这个数值可能需要调整）
        # 因此，输入到第一个全连接层的特征数量是n_qubits * 6

        # 第一个全连接层，输入特征数为n_qubits * 6，输出特征数为
        self.fc1 = nn.Linear(4, 450)
        self.fc2 = nn.Linear(450, 64)
        self.fc3 = nn.Linear(64, 49)
        self.fc4 = nn.Linear(49, 3)

    def forward(self, x):
        bs = x.shape[0]
        # 假设输入x的形状是(bs, c, h, w)，并且已经通过Quanv2d类处理成适合全连接层输入的形状
        x = x.view(bs, 1, 3, 3)  # 根据Quanv2d类的输出调整这个形状
        x = self.qc(x)

        # 将量子层的输出展平成一维向量，以便输入到全连接层
        x = x.view(x.size(0), -1)

        # 通过全连接层
        x = self.fc1(x)
        # X = self.lr1(X)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


