# Probability-based-Edge-Detection

Course Homework for RBE549 - Computer Vision (Spring 2024)

## Requirements:

1. CUDA Toolkit + GPU drivers

2. Pytorch

3. Numpy

4. Matplotlib

5. Opencv

### Phase 1:

1. Navigate to 'pshinde1_hw0/Phase1/Code/'.
2. Run 
```
python3 Wrapper.py
```
3. The filter banks will be saved to FilterBanks folder.
4. The image outputs will be saved to ImageRes folder.

### Phase 2:

#### Training:
1. To start training, go to 'pshinde1_hw0/Phase2/Code/'
2. To implement Basic Network, run the following
```
python3 Train.py --NetNum 1
```
3. To implement Residual Network, run the following
```
python3 Train.py --NetNum 2
```
4. To implement Resdiual NeXt Network, run the following
```
python3 Train.py --NetNum 3
```
5. To implement Dense Network, run the following
```
python3 Train.py --NetNum 4
```
6. The Logs and Checkpoints get saved to '{NetworkName}/Logs/' and '{NetworkName}/Checkpoints/' respectively. You need provide the paths to save Logs and Checkpoints. To do so, run the following 
```
python3 Train.py --NetNum {} --CheckPointPath {} --LogsPath {}
```

#### Testing:
1. To test Basic Network, provide the ModelPath for the latest saved checkpoint for that model and  run the following
```
python3 Test.py --NetNum 1 --ModelPath ./Checkpoints/BasicNet/14model.ckpt
```
Uncomment line 275 in Train.py and line 230 in Test.py to implement the chosen transforms for this network. Additionally, use SGD as the optimizer.

2. To test Basic Network (Improved), provide the ModelPath for the latest saved checkpoint for that model and  run the following
```
python3 Test.py --NetNum 1 --ModelPath ./Checkpoints/BasicNet/19model.ckpt
```
This network uses ToTensor() as transform. Comment the lines mentioned previously to implement improved network. Additionally, use AdamW as the optimizer.

3. To test Residual Network, provide the ModelPath for the latest saved checkpoint for that model and run the following
```
python3 Test.py --NetNum 2 -ModelPath ./Checkpoints/ResNet/19model.ckpt
```
4. To implement Resdiual NeXt Network, provide the ModelPath for the latest saved checkpoint for that model and run the following
```
python3 Test.py --NetNum 3 -ModelPath ./Checkpoints/ResNeXt/14model.ckpt
```
5. To implement Dense Network, provide the ModelPath for the latest saved checkpoint for that model and run the following
```
python3 Test.py --NetNum 4 -ModelPath ./Checkpoints/DenseNet/14model.ckpt
```
6. Number of parameters, Test Accuarcy, Train ACcuracy and Confusion Matrices will be displayed as the output for this script.
