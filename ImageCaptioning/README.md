# Image Captioning
Using Resnet-152 to encode the image and using RNN(LSTM based) to output the caption
![alt text](png/model.png)

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py#L37). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a [for-loop](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L48).



## Step by step usage
#### 1. Install Dependencies
##### a. Install Pycocotools
```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
```

##### b. Install requirements.txt
```bash
pip install -r requirements.txt
```

##### c. Download the COCO Dataset
```bash
pip install -r requirements.txt
```

#### 2. Download the dataset
First, download the annotations for COCO Captions train and val 2014
```bash
chmod +x download.sh
./download.sh
```

#### 3. Preprocessing:
Build Vocabulary (word and idx mapping) and image resize for training. By default it build vocab for every token (using NLTK Tokenizer) when a token appear >= 4 (from COCO Captions) and resize all train images to 256 x 256
```bash
python build_vocab.py   
python resize_train_image.py
```

#### 4. Train the model (With mock Federated Learning Applied)
```bash
python train.py    
```

#### 5. Test the model (With Quantization applied)

```bash
python inference.py --image='png/example.png'
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and the vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0). You should extract pretrained_model.zip to `./models/` and vocab.pkl to `./data/` using `unzip` command.
