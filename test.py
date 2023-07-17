import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import Model
import Dataset
import numpy as np
import logging
from datetime import datetime


testdataset = Dataset.Dataset_stations_all(np.load('sh_data.npy')[55:,:,:, :],np.load('weather.npy')[55:])
device = torch.device('cuda')
best_model_path='model\\cnn_gnn_best_model.pth'
batch_size = 8
model=Model.MyCNNandGNN2().to(device)
testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
model.load_state_dict(torch.load(best_model_path))
test_loss = torch.tensor([0., 0., 0., 0.]).to(device)
test_rmseloss = torch.tensor([0., 0., 0., 0.]).to(device)
log_file = f'log\\cnn_gnn_test_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
model.eval()
with torch.no_grad():
    for inputs, weather,q, labels in testdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        weather = weather.to(device)
        q=q.to(device)
        outputs = model.forward(inputs,  adj_matrix3, weather,q)
        loss = torch.abs(outputs - labels).mean(dim=[0, 1,2])
        loss2=torch.square(outputs - labels).mean(dim=[0, 1,2])
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
