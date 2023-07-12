import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MyCNN2(nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 18, 128)
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(136, 64)
        self.out_fc=nn.Linear(64,8)

    def forward(self, x,q):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x=torch.concat([x,q],dim=-1)
        x =  nn.functional.relu(self.fc2(x))
        x= self.out_fc(x).reshape(-1,4,2)
        return x

class CNN_feature2(nn.Module):
    def __init__(self):
        super(CNN_feature2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 18, 120)

    def forward(self, x,q):
        '''
        x:(b*288,2,73,14);
        q:(b*288,8);
        return (b*288,120)
        '''
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.concat([x,q],dim=1)
        
class GNN(nn.Module):
    def __init__(self,input_size):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_size,input_size)

    def forward(self, x, adj_matrix):
        '''
        x:(b,nun_node,node_feature);
        adj_matrix:(num_node,num_node);
        return (b,nun_node,node_feature);
        '''
        adj_matrix=adj_matrix.repeat(x.size(0),1,1)
        x = torch.bmm(adj_matrix, x)
        x = self.fc1(x)
        x=nn.functional.relu(x)
        return x
    
class MyCNNandGNN2(nn.Module):
    def __init__(self):
        super(MyCNNandGNN2, self).__init__()
        self.cnn = CNN_feature2()
        self.gnn1 = GNN(128)
        self.gnn2 = GNN(128)
        self.gnn3 = GNN(128)
        self.weather_embedding=nn.Embedding(2,2)
        self.fc1 = nn.Linear(128*4, 128)
        self.fc2 = nn.Linear(128,8)
    def forward(self, x,q, adj_matrix1, adj_matrix2, adj_matrix3):
        '''
        x:(b,288,2,14,73);
        q:(b,288,2,4)
        adj_matrix:(288,288)
        '''
        batch_size,station_num,day_num,time_num = x.size(0),x.shape[1],x.shape[3],x.shape[4]
        x = x.view(batch_size * station_num, 2, day_num, time_num)
        cnn_out = self.cnn.forward(x,q.reshape(batch_size*station_num,8)).reshape(batch_size,station_num,128)
        gnn_out1 = self.gnn1(self.gnn1(cnn_out, adj_matrix1),adj_matrix1)
        gnn_out2 = self.gnn2(self.gnn2(cnn_out, adj_matrix2),adj_matrix2)
        gnn_out3 = self.gnn3(self.gnn3(cnn_out, adj_matrix3),adj_matrix3)   #(b,station_num,128)
        
        x =torch.cat([gnn_out1,gnn_out2,gnn_out3,cnn_out],dim=-1)

        x = self.fc1(x)
        x=self.fc2(nn.functional.relu(x))
        return x.reshape(batch_size,station_num,2,4)
