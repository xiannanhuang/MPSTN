# MPSTN：Leveraging Intra-Period and Inter-Period Features for Enhanced Urban Rail Passenger Flow Prediction
Code for paper 'Leveraging Intra-Period and Inter-Period Features for Enhanced Urban Rail Passenger Flow Prediction'.<br /> 
We provide our trained model in model//cnn_gnn_best_model.pth 
## Abstract
Accurate short-term prediction of passenger flow in subway stations plays a vital role in enabling subway station personnel to proactively address changes in passenger volume, thus facilitating the development of intelligent transportation systems. Despite existing literature in this field, there is a lack of research on effectively integrating features from different periods, particularly intra-period and inter-period features, for subway station flow prediction. In this paper, we propose a novel model called Muti Period Spatial Temporal Net (MPSTN) that leverages features from different periods by transforming one-dimensional time series data into two-dimensional matrices based on periods. The folded matrices exhibit structural characteristics similar to images, enabling the utilization of image processing techniques, specifically convolutional neural networks (CNNs), to integrate features from different periods. Therefore, our model incorporates a CNN module to extract temporal information from different periods and a graph neural network (GNN) module to integrate spatial information from different stations. We compared our approach with various state-of-the-art methods for spatiotemporal data prediction using a publicly available dataset and achieved minimal prediction errors.

## Train

python cnn_gnn2_mian.py

## Test
python test.py

## Requirements
python 3.7 <br />
torch  1.13.0+cu117
