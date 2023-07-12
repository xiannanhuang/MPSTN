import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import Model
import Dataset
import numpy as np
import logging
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import random
# 定义学习率衰减策略

import matplotlib.pyplot as plt
dataset = Dataset.Dataset_as_cv2(np.load('sh_data.npy')[:62, :, :, :],14)
validdataset = Dataset.Dataset_as_cv2(np.load('sh_data.npy')[48:69 ,:,:, :],14)
testdataset = Dataset.Dataset_as_cv2(np.load('sh_data.npy')[55:,:,:, :],14)
# validdataset = Dataset.Dataset_stations_all(np.delete(np.load('sh_metro.npy'), np.s_[-17:-10], axis=0)[-28:, :,:, :])
device = torch.device('cuda')
# 创建数据加载器
batch_size = 2048
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validdataloader = DataLoader(validdataset, batch_size=batch_size, shuffle=False)
testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
log_file = f'log\\CNN2_training_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'

logging.basicConfig(filename=log_file, level=logging.INFO)
model=Model.MyCNN2().to(device)
# 定义损失函数和优化器
criterion = torch.nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=2, gamma=0.95)
# 训练模型
num_epochs = 300
# model.load_state_dict(torch.load('F:\subway_short_pre\\fina_model_0622\model\\best_model_20230703161612.pth'))
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(f'{timestamp} -' + 'train start for model cnn2')
# adj_matrix1=torch.tensor(np.load('graph_sh_sml.pkl',allow_pickle=True),dtype=torch.float32).to(device)
# adj_matrix2=torch.tensor(np.load('graph_sh_cor.pkl',allow_pickle=True),dtype=torch.float32).to(device)
# adj_matrix3=torch.tensor(np.load('graph_sh_conn.pkl',allow_pickle=True),dtype=torch.float32).to(device)
l1_lambda = 0.00
# 记录网络各层参数的名字和形状
for name, param in model.named_parameters():
    logging.info(f"Parameter Name: {name}\t Shape: {param.shape}")

# 计算并记录可训练的参数总量
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total Trainable Parameters: {total_trainable_params}")

# 记录当前程序的各个参数的名称和数值
logging.info(f"num_epochs: {num_epochs}")
logging.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
best_test_loss = float('inf')  # 初始化最小测试误差为正无穷大
for epoch in range(num_epochs):
    epoch_start_time=datetime.now()
    model.train()
    running_loss = 0.0
    i = 0
    for inputs,q, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        q=q.to(device)


        # 清除优化器梯度
        optimizer.zero_grad()

        # 前向传播

        outputs = model.forward(inputs,q)
        # 计算损失
        loss = criterion(outputs, labels)



        # 反向传播和更新模型参数
        loss.backward()
        optimizer.step()

        # 累计损失
        running_loss += loss.item()
        i += 1

        # 打印训练进度
        if (i + 1) % 100 ==0:
            avg_loss = running_loss / 100
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                          len(dataloader), avg_loss)
            print(log_msg)
            logging.info(f'{timestamp} - {log_msg}')
            running_loss = 0.0
    model.eval()
    with torch.no_grad():
        val_loss = torch.tensor([0.,0.,0.,0.]).to(device)
        for inputs,q,labels in validdataloader:
            inputs = inputs.to(device)
            q=q.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs,q)
            loss = torch.abs(outputs-labels).mean(dim=[0,-1])
            val_loss += loss
        avg_val_loss = val_loss / len(validdataloader)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = 'Validation Loss 15min: {:.4f};Validation Loss 30min: {:.4f};Validation Loss 45min: {:.4f};Validation Loss 60min: {:.4f}'.format(avg_val_loss[0],avg_val_loss[1],avg_val_loss[2],avg_val_loss[3])
        print(log_msg)
        logging.info(f'{timestamp} - {log_msg}')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_duration = datetime.now() - epoch_start_time
        log_msg = f"Epoch: {epoch+1}\tLearning rate: {optimizer.param_groups[0]['lr']:.6f}\tDuration: {epoch_duration}"
        print(log_msg)
        logging.info(f"{timestamp} - {log_msg}")
    scheduler.step()
    test_loss = torch.tensor([0., 0., 0., 0.]).to(device)
    test_rmseloss = torch.tensor([0., 0., 0., 0.]).to(device)
    model.eval()
    with torch.no_grad():
        for inputs, q, labels in testdataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            q = q.to(device)
            outputs = model.forward(inputs, q)
            loss = torch.abs(outputs - labels).mean(dim=[0, -1])
            loss2=torch.square(outputs - labels).mean(dim=[0, -1])
            test_loss += loss
            test_rmseloss+=loss2

    avg_test_loss = test_loss / len(testdataloader)
    rmse_test_loss =torch.sqrt(test_rmseloss / len(testdataloader))
    # Calculate RMSE loss

    log_msg = f"Test Loss 15min: {avg_test_loss[0]:.4f}\tTest Loss 30min: {avg_test_loss[1]:.4f}\tTest Loss 45min: {avg_test_loss[2]:.4f}\tTest Loss 60min: {avg_test_loss[3]:.4f}"
    logging.info(log_msg)
    print(log_msg)
    rmse_log_msg = f"RMSE Loss 15min: {rmse_test_loss[0]:.4f}\tRMSE Loss 30min: {rmse_test_loss[1]:.4f}\tRMSE Loss 45min: {rmse_test_loss[2]:.4f}\tRMSE Loss 60min: {rmse_test_loss[3]:.4f}"
    logging.info(rmse_log_msg)
    print(rmse_log_msg)
    if avg_test_loss.sum() < best_test_loss:
        best_test_loss = avg_test_loss.sum()
        best_model_name = f"model\\cnn2_ best_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
        torch.save(model.state_dict(), best_model_name)
        logging.info(f"Saved best model: {best_model_name}")
# 生成带有当前时间的模型名称
model_name = f"cnn2_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"

# 保存模型
torch.save(model.state_dict(), model_name)

# 打印存储的模型名称
logging.info(f"Saved model: {model_name}")
