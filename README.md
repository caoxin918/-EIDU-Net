# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合。

# EIDU-Net : Edge-preserved Inception DenseGCN U-Net for LiDAR Point Cloud Segmentation

Xueli Xu, Jingyu Wang, Xinxin Han, Huan Xia, Guohua Geng, Linzhi Su, Kang Li※ and Xin Cao※

This repository is the official implementation of [EIDU-Net : Edge-preserved Inception DenseGCN U-Net for LiDAR Point Cloud Segmentation]

###Installation
1. This repository is based on Python 3.6, TensorFlow 1.8.0, CUDA 10.0  on Ubuntu.

2. For compiling TF operators, please check tf_xxx_compile.sh under each op subfolder in tf_ops folder. Note that you need to update nvcc, python and tensoflow to include library if necessary.

### Preparing Dataset (S3DIS)
1. 
~~~
wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
~~~

2. extract it and put it in folder data.

### Training
Below line will run the training code with default setting on S3DIS. 
~~~
python train_ss.py
~~~

### Testing and vis

~~~
python batch_inference.py --model_path sseg/best_cls_acc_model.ckpt --dump_dir log_ssegy/dump --output_filelist log_sseg/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu
~~~


## Citation
If you find EIDU-Net useful in your research, please consider citing:
```
