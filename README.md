# Image Captioning
The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used resnet-152 model pretrained on the ILSVRC-2012-CLS image classification dataset. The decoder is a long short-term memory (LSTM) network.

![image](https://github.com/bhavanap12/image_caption_generator/assets/23119773/44ded050-dda4-40b2-a2cd-ac87d10cd48c)


## Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is "Giraffes standing next to each other", the source sequence is a list containing ['<start>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other'] and the target sequence is a list containing ['Giraffes', 'standing', 'next', 'to', 'each', 'other', '<end>']. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

## Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using encoder.eval(). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a for-loop.

## Usage
1. Download the dataset
pip install -r requirements.txt
chmod +x download.sh
./download.sh
2. Preprocessing
python build_vocab.py   
python resize.py
3. Train the model
python train.py    
4. Test the model
python sample.py --image=<image-file-path>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model here and the vocabulary file here. You should extract pretrained_model.zip to ./models/ and vocab.pkl to ./data/ using unzip command.

## Examples
This is the home page
![Screenshot 2024-01-22 190835](https://github.com/bhavanap12/image_caption_generator/assets/23119773/44d176ed-624e-4b78-82b8-14ad1e74760d)

The image is displayed after it's uploaded
![Screenshot 2024-01-22 190901](https://github.com/bhavanap12/image_caption_generator/assets/23119773/714e160f-9a9b-4382-bef2-d6ce7b3b8f43)




Caption is generated once we click on Generate Caption
![Screenshot 2024-01-22 190909](https://github.com/bhavanap12/image_caption_generator/assets/23119773/601c0870-e735-41d6-a976-424ba106068d)



![Screenshot 2024-01-22 191012](https://github.com/bhavanap12/image_caption_generator/assets/23119773/bdc7be2d-94ad-4fa5-8cf1-ad29afadca42)
