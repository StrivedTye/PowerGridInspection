# Power Grid Inspection from LiDAR Point clouds

There are some tasks in power grid inspection, including semantic segmentation, instance segmentation...

## Setup
### Dependencies
Here we list the most important part of our dependencies

| Dependency        | Version                    |
| ----------------- | -------------------------- |
| open3d            | 0.15.2                     |
| python            | 3.8.0                      |
| pytorch           | 1.13.1(cuda11.7,cudnn8) |
| pytorch-lightning | 1.5.10                     |
| pytorch3d         | 0.7.5                      |
| shapely           | 2.0.0                      |
| laspy             | 2.5.4                     |


## Dataset
### powerUAV
The format of point clouds is `.las`. 
We may need to pre-process these data. 


## Get Started
### Training

To train a model, you must specify the `.yaml` file. The `.yaml` file contains all the configurations of the dataset and the model. We provide `.yaml` files under the [configs/](./configs) directory. 

**Note:** Before running the code, you will need to edit the `.yaml` file.

```
python main.py configs/seg_RandLANet_poweruav_cfg.yaml --gpus 0 1
```

### Testing

To test a trained model, specify the checkpoint location with `--resume_from` argument and set the `--phase` argument as `test`.

```
python main.py configs/mbptrack_kitti_ped_cfg.yaml --phase test --resume_from pretrained/mbptrack_kitti_ped.ckpt
```



