import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from app.load_data import MyCSVDatasetReader as CSVDataset
#from models.classical import Net
from models.single_encoding import Net
#from models.multi_encoding import Net
#from models.hybrid_layer import Net
#from models.inception import Net
#from models.multi_noisy import Net
from app.train import train_network

# load the dataset
#./datasets/mnist_179_1200.csv
#./datasets/features.csv
dataset = CSVDataset('./datasets/Breast Cancer Wisconsin.csv')
#dataset = CSVDataset('./datasets/mnist_179_1200.csv')
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'


# load the device
device = torch.device('cpu')

# define model
net = Net()
# net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.32)  # 例如，使用Adam优化器
#optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)  # 使用SGD优化器，并设置学习率和动量

epochs = 100
bs = 4
print(len(dataset))
print(list(range(len(dataset))))
train_id, val_id = train_test_split(list(range(len(dataset))), test_size = 0.2, random_state = 0)
train_set = Subset(dataset, train_id)
val_set = Subset(dataset, val_id)
print(train_set)
print(len(train_id),len(val_id))
train_network(net = net, train_set = train_set, val_set = val_set, device = device,
epochs = epochs, bs = bs, optimizer = optimizer, criterion = criterion)  # outdir = outdir, file_prefix = file_prefix)
