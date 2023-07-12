import torch
import torch.nn as nn

import torch
import torch.nn as nn

# class MyCNN(nn.Module):
#     def __init__(self):
#         super(MyCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 3 * 18, 128)
#         self.relu3 = nn.ReLU()
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 4)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x =  (self.fc2(x))
#         return x

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

# class Mlp_stations(nn.Module):
#     def __init__(self):
#         super(Mlp_stations, self).__init__()
#         self.fc1 = nn.Linear(4 * 288 * 2, 512)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(2048, 2048)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(512, 1024)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(1024, 4 * 288 * 2)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         # x = self.fc2(x)
#         # x = self.relu2(x)
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.fc4(x)
#         return x
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class MyTransformer(nn.Module):
#     def __init__(self, input_dim=8, hidden_dim=64, output_dim=8, num_layers=2):
#         super(MyTransformer, self).__init__()
#         self.token_projection = nn.Linear(input_dim, hidden_dim)
#         self.transformer_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
#         self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
#         self.output_projection = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.token_projection(x)
#         x = x.permute(1, 0, 2)  # 将维度变为(seqlen, batch_size, hidden_dim)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # 将维度变回(batch_size, seqlen, hidden_dim)
#         x = self.output_projection(x)
#         return x
# class MyCNN4transformer(nn.Module):
#     def __init__(self):
#         super(MyCNN4transformer, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 3 * 18, 256)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

# class MyCnnandTransformer(nn.Module):
#     def __init__(self, hidden_dim=128, num_layers=1):
#         super(MyCnnandTransformer, self).__init__()
#         self.cnn_model = MyCNN4transformer()
#         self.transformer_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4,dropout=0.5)
#         self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
#         self.output_projection = nn.Linear(hidden_dim, 4)

#     def forward(self, x):
#         ha_pred = (x[:, :, -7, :4] + x[:, :, -14, :4]) / 2 ##x batchsize*576*14*73
#         x=x.reshape(-1,1,14,73)
#         cnn_out = self.cnn_model(x).reshape(-1,288*2,128)
#         x = self.transformer_encoder(cnn_out)
#         x = x+cnn_out  # 将维度变回(batch_size, hidden_dim, seq_len)
#         # x=cnn_out
#         x = self.output_projection(x)
#         return x
# class CNN_feature(nn.Module):
#     def __init__(self):
#         super(CNN_feature, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 3 * 18, 256)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x
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

# class MyCNNandGNN2(nn.Module):
#     def __init__(self):
#         super(MyCNNandGNN2, self).__init__()
#         self.cnn = CNN_feature()
#         self.gnn1 = GNN(256)
#         self.gnn2 = GNN(256)
#         self.gnn3 = GNN(256)
#         self.weather_embedding=nn.Embedding(2,2)
#         self.fc1 = nn.Linear(130, 32)
#         self.fc2 = nn.Linear( 32,4)
#     def forward(self, x, adj_matrix1, adj_matrix2, adj_matrix3,weather):
#         ha_pred = ((x[:, :,:, -7, :4] + x[:, :,:, -14, :4]) / 2 ).reshape(-1,576,4)
#         batch_size = x.size(0)
#         weather_emb=self.weather_embedding(weather).reshape(batch_size,1,2).repeat(1,576,1)
#         x = x.view(batch_size * 288, 2, 14, 73)
#         cnn_out = self.cnn(x)
#         x=cnn_out.view(batch_size,288,256)
#         # x=torch.concat([x,weather_emb],dim=-1)
#         gnn_out1 = self.gnn1(x, adj_matrix1)
#         gnn_out2 = self.gnn2(x, adj_matrix2)
#         gnn_out3 = self.gnn3(x, adj_matrix3)
        
#         x =x+ gnn_out1 + gnn_out2 + gnn_out3

#         x = self.fc1(torch.concat([x.view(batch_size,288,2,128).view(batch_size,576,128),weather_emb],dim=-1))
#         x=self.fc2(nn.functional.relu(x))
#         return x
# class MyCNNandGNN2(nn.Module):
#     def __init__(self):
#         super(MyCNNandGNN2, self).__init__()
#         self.gnn1 = GNN(32,64)
#         self.gnn2 = GNN(32,64)
#         self.gnn3 = GNN(32,64)
#         self.gnn21 = GNN(64,64)
#         self.gnn22 = GNN(64,64)
#         self.gnn23 = GNN(64,64)
#         self.fc = nn.Linear(128, 4)
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 3 * 18, 8)
#         self.relu3 = nn.ReLU()

#     def forward(self, x, adj_matrix1, adj_matrix2, adj_matrix3):
#         ha_pred=((x[:,:,:,-7,:4]+x[:,:,:,-14,:4])/2).reshape(-1,288*2,4)
#         batch_size = x.size(0)
#         x = x.view(batch_size * 288, 2, 14, 73)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)  #(batchsize*288)*64*w*h
#         w,h=x.size(2),x.size(3)
#         x=x.reshape(batch_size,288,32,w,h).permute(0,3,4,1,2).reshape(-1,288,32)
    
        
#         gnn_out1 = self.gnn1(x, adj_matrix1)
#         gnn_out2 = self.gnn2(x, adj_matrix2)
#         gnn_out3 = self.gnn3(x, adj_matrix3)
        
#         x = gnn_out1 + gnn_out2 + gnn_out3+x
#         x = x.reshape(batch_size,w,h,288,32).permute(0,3,4,1,2).reshape(batch_size*288,32,w,h)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         w,h=x.size(2),x.size(3)
#         x=x.reshape(batch_size,288,64,w,h).permute(0,3,4,1,2).reshape(-1,288,64) ##(batchsize*288)*w*h*64
    
        
#         gnn_out1 = self.gnn21(x, adj_matrix1)
#         gnn_out2 = self.gnn22(x, adj_matrix2)
#         gnn_out3 = self.gnn23(x, adj_matrix3)
        
#         x = gnn_out1 + gnn_out2 + gnn_out3+x
#         x = x.reshape(batch_size,w,h,288,64).permute(0,3,4,1,2).reshape(batch_size*288,64,w,h)
#         x =  (self.fc1(x.reshape(batch_size,288,-1))).reshape(batch_size,288*2,4)+ha_pred
        
#         return x
# 定义整个模型
# class MyCNNandGNN(nn.Module):
#     def __init__(self):
#         super(MyCNNandGNN, self).__init__()
#         self.cnn = CNN_feature()
#         self.gnn1 = GNN(258)
#         self.gnn2 = GNN(258)
#         self.gnn3 = GNN(258)
#         self.weather_embedding=nn.Embedding(2,2)
#         self.fc1 = nn.Linear(129, 32)
#         self.fc2 = nn.Linear(32,4)
#     def forward(self, x, adj_matrix1, adj_matrix2, adj_matrix3,weather):
#         batch_size = x.size(0)
#         weather_emb=self.weather_embedding(weather).reshape(batch_size,1,2).repeat(1,288,1)
#         x = x.view(batch_size * 288, 2, 14, 73)
#         cnn_out = self.cnn(x)
#         x=cnn_out.view(batch_size,288,256)
#         x=torch.concat([x,weather_emb],dim=-1)
#         gnn_out1 = self.gnn1(x, adj_matrix1)
#         gnn_out2 = self.gnn2(x, adj_matrix2)
#         gnn_out3 = self.gnn3(x, adj_matrix3)
        
#         x =x+ gnn_out1 + gnn_out2 + gnn_out3

#         x = self.fc1(x.view(batch_size,288,2,129).view(batch_size,576,129))
#         x=self.fc2(nn.functional.relu(x))
#         return x