# R&D center WINSTARS.AI 

## Test task in Data Science

---

## Quick start
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download model and run training:

```bash
bash scripts/download_data.sh
python train.py --amp
```


```bash
python download_data.py 
```

## Usage
**Note : Use Python 3.11 or newer**

### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg --viz`



```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--viz] [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --viz, -v             Visualize the images as they are processed
  
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.

## Data
The AirBus Ship Detection data is available on the [Kaggle website](https://www.kaggle.com/competitions/airbus-ship-detection/data).

## Complete description of solution

**Task**

I have never worked with CV problems except for MNIST classification, so It was challenging to understand what my task was. I had to check other Kaggler's solutions. Also, it's the first time I've seen RLE decoding. 

**EDA**

As soon as I understood my task, I started the EDA phase. I observed that DataSet is unbalanced. 78% of images have no ships on them. Also, those ships are too small, 0.05% of pixels are related to ships. This imbalance makes creating a good model - a challenging task. 

**Data Preprocessing**

The major problem in this phase was creating a representative data set for training. I have decided to drop all non-ship images. Also, the memory issue I faced was fixed by splitting the data into batches of 64 images. Then bathes are converted into small numpy arrays with shape [768,768,1] for masks and [768,768,3] - for images. Then all these batches were merged into two big arrays. In the end, I created a custom PyTorch DataSet 

**Training Model**

I have never worked with TensorFlow, so I decided to make a good training process using PyTorch. The default Unet model was chosen, because it's the simplest CNN model for semantic segmentation problems.

**Evaluation** 

In the technical task were suggestions to use the Dice score, but LoU is used as a metric by Airbus. So I decided to use Dice. 

**Results**

I had limited time to fine-tune the model because most of the time I have spent on EDA and Data Preprocessing. I have seen the solution where people were using transfer learning. Then trained the classification model, and used those weights for semantic segmentation, I wanted to try, but the lack of time didn't leave a chance. 

Even so, I have installed CUDA and 3070Ti GPU - it takes a significant amount of time, to train a good model, so as an example that my model is working I trained it on 1k images(900 for training, 100 evaluation). There are images in the dir folder that show that on even that small Data, the model can detect ships on images.

---


Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)