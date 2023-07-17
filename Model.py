import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CNN_feature(nn.Module):
    def __init__(self):
        super(CNN_feature, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3456, 246)

    def forward(self, x):
        '''x:(b*288,2,73,14);
        return:(b*station_num,128)
        '''
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class GNN_layer(nn.Module):
    def __init__(self,input_size):
        super(GNN_layer, self).__init__()
        self.fc1 = nn.Linear(input_size,input_size)

    def forward(self, x, adj_matrix):
        '''
         x: (batch_size, num_nodes, node_feature)
        adj_matrix: (num_nodes, num_nodes)
        return: (batch_size, num_nodes, node_feature)'''
        adj_matrix = self.normalize_adj_matrix(adj_matrix)
        adj_matrix = adj_matrix.repeat(x.size(0), 1, 1)
        x = torch.bmm(adj_matrix, x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return x

    def normalize_adj_matrix(self, adj_matrix):
        '''
        Normalize the adjacency matrix.
        adj_matrix: (num_nodes, num_nodes)
        return: (num_nodes, num_nodes)
        '''
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        degree_matrix_sqrt_inv = torch.sqrt(torch.inverse(degree_matrix))
        normalized_adj_matrix = torch.matmul(torch.matmul(degree_matrix_sqrt_inv, adj_matrix), degree_matrix_sqrt_inv)
        return normalized_adj_matrix

class MyCNNandGNN2(nn.Module):
    def __init__(self):
        super(MyCNNandGNN2, self).__init__()
        self.gnn_layer_num=2
        self.cnn = CNN_feature()
        self.gnn = [GNN_layer(256).cuda() for _ in range(self.gnn_layer_num)]
        self.weather_embedding=nn.Embedding(2,2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear( 128,8)
    def forward(self, x, adj_matrix,weather,q):
        '''
        x:(b,288,2,14,73);
        adj_matrix:(1,288,288)
        q:(b,288,2,4)
        weather:(b)
        '''
        batch_size,station_num,day_num,time_num = x.shape[0],x.shape[1],x.shape[3],x.shape[4]
        weather_emb=self.weather_embedding(weather).reshape(batch_size,1,2).repeat(1,station_num,1)
        x=x.view(batch_size * station_num, 2, day_num, time_num)
        cnn_out = torch.cat([self.cnn(x).reshape(batch_size,station_num,246),q.reshape(batch_size,station_num,8),weather_emb],dim=-1)
        # x=torch.concat([x,weather_emb],dim=-1)
        gnn_out=cnn_out.clone()
        for i in range(self.gnn_layer_num):
            gnn_out=self.gnn[i](gnn_out,adj_matrix)
        x=torch.cat([gnn_out,cnn_out],dim=-1)

        x = self.fc1(x)
        x=self.fc2(nn.functional.relu(x)).reshape(batch_size,station_num,2,4)
        return x
