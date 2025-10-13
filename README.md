## 1 Dataset Preparation

## 1.1 GoPro dataset
1. To download GoPro training and testing data, run
```
 python download_data.py --data train-test
```
2. Generate image patches from full-resolution training images of GoPro dataset
   
```
python generate_patches_gopro.py 
```
## 1.2 QR code dataset

1. Prepare you QR code and generate motion blur dataset, run
```
python motionblur/generate_dataset.py 
``` 

2. Generate image patches from full-resolution training images of QR code dataset

``` 
python generate_patches_qrcode.py 
``` 

## 2 Training

## 2.1 EG-Restormer

1. To train EGRestormer Training(pretaining) on GoPro dataset, run
   
```
cd ADNet

torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_port=4321 basicsr/train.py -opt options/train/train_egrestormer.yml --launcher pytorch
```

**Note:** The above training script uses 6 GPUs by default. To use any other number of GPUs, modify [options/train/train_egrestormer.yml](options/train/train_egrestormer.yml)


2. Training (finetuning) on QRData dataset, run
```  
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_port=4321 basicsr/train.py -opt options/train/train_egrestormer_qrdataset.yml --launcher pytorch
```

**Note:** The training iteration start from the iteration pretained on GoPro, you can modify i [options/train/train_egrestormer_qrdataset.yml](options/train/train_egrestormer_qrdataset.yml)


## 2.2 LENet

1. To train LENet on Gopro, run
   
```
cd ADNet
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_port=4321 basicsr/train.py -opt options/train/train_lenet.yml --launcher pytorch
```

2. Training (finetuning) on QRData dataset, run
```  
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_port=4321 basicsr/train.py -opt options/train/train_lenet.yml --launcher pytorch
``` 

**Note:** change the dataset in the training script.



## 3 Evaluation

#### 3.1 Download the trained on GoPro only EG-Restormer model

Download the pre-trained EG-Restormer [gopro_pretrain_model](https://drive.google.com/file/d/15fHBJy2MgMy2pcCKGYCzXpCHO_3Ja-O7/view?usp=sharing) and place it in `./experiment/egrestormer/models/`

#### 3.2 Download the trained on GoPro + QRData EG-Restormer model

Download the pre-trained EG-Restormer [QRData_model](https://drive.google.com/file/d/12NCPyL2lvga3z85WRjc-LzZb18qcUbgY/view?usp=sharing) and place it in `./experiment/egrestormer/models/`

#### 3.3 Download the LENet model

Download the pre-trained EG-Restormer [QRData_model](https://drive.google.com/file/d/1foqYyFqTLhOL10Dt3u8b9XDhaPdaxcEr/view?usp=sharing) and place it in `./experiment/LENet/models/`


#### 3.4 Testing on QR code dataset

- Download QR code test set [QR code](https://drive.google.com/file/d/1B49u5Rf0RgGbsyaHMYEZXBBG_NC1h14M/view?usp=sharing) and place it in `./datasets/test/`


- Testing **EG-Restormer**, run,

```
python test_egrestormer.py
```

- Testing **LENet**, run,

```
python test_lenet.py
```

- Testing **ADNet**, run,

```
python ADNet.py
```

## Contact
If you have any question, please contact ljphit@163.com

**Acknowledgment:** This code is based on the [Restormer](https://github.com/swz30/Restormer/tree/main) and [NAFNet](https://github.com/megvii-research/NAFNet/tree/main)