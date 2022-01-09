# PILOT with multi-shot

## Usage
### Training
The training script will train the model using the train and validation datasets. 
The saved model and the tensorboard logs will be save in `summary/{test-name}`. 
```
CUDA_VISIBLE_DEVICES=0 python train.py --test-name 'test_name' --n-shots={n_shots} --initialization={init} 
                             --trajectory-learning={trajectory_learning} --sub-lr={sub_lr} --lr={rec_lr} 
```
Where:
- `test-name`: Name of the experiment.  
- `n-shots`: Number of shots.  
- `initialization`: Initialization of the trajectory, one of 'cartesian'/'radial'.  
- `trajectory-learning`: If set to True will learn both the reconstruction network and the trajectory, else will learn only the reconstruction.  
- `sub-lr`: learning rate of the subsampling layer.  
- `lr`: Learning rate of the reconstruction network.  
 
Full list of possible arguments can be seen in `train.py`.  

## Dependencies
You need first to install all the dependencies using:
```
pip install -r req.txt
```
I guess that we can upgrade to newer versions of must of this libraries.