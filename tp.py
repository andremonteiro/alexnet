
# coding: utf-8

# # Introduction
# In this notebook we will test the implementation of the AlexNet class provided in the `alexnet.py` file. This is part of [this](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html) blog article on how to finetune AlexNet with TensorFlow 1.0.
# 
# To run this notebook you have to download the `bvlc_alexnet.npy` file from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/), which stores the pretrained weigts of AlexNet.
# 
# The idea to validate the implementation is to create an AlexNet graph with the provided script and load all pretrained weights into the variables (so no finetuneing!), to see if everything is wired up correctly.

# In[1]:


#some basic imports and setups
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
#image_dir = os.path.join(current_dir, 'images')
image_dir = "/home/andre/Downloads/celeba/img_align_celeba";

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

print("Carregando imagens...");

#get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
img_files.sort();


print("Imagens Carregadas");


#load all images
#imgs = []
#for f in img_files:
#    imgs.append(cv2.imread(f))
    
# First we will create placeholder for the dropout rate and the inputs and create an AlexNet object. Then we will link the activations from the last layer to the variable `score` and define an op to calculate the softmax values.

# In[2]:


from alexnet import AlexNet
from caffe_classes import class_names

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000, [])

#define activation of last layer as score
score = model.fc8
fc7 = model.fc7
conv1 = model.norm1
conv5 = model.pool5

#create op to calculate softmax 
softmax = tf.nn.softmax(score)


# Now we will start a TensorFlow session and load pretrained weights into the layer weights. Then we will loop over all images and calculate the class probability for each image and plot the image again, together with the predicted class and the corresponding class probability.

# In[3]:

with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load the pretrained weights into the model
    model.load_initial_weights(sess)
    
    # Create figure handle
    fig2 = plt.figure(figsize=(15,6))
    
    # Loop over all images
    for i, image in enumerate(img_files):  
        print("Extraindo "+img_files[i]+". Restam "+str((len(img_files)-i)));

        filename = os.path.split(img_files[i])[1]
        filename = os.path.splitext(filename)[0]
        dir = 'features/'+filename+'/';
        if not os.path.exists(os.path.dirname(dir)):
            image = cv2.imread(image);                  
            # Convert image to float32 and resize to (227x227)
            img = cv2.resize(image.astype(np.float32), (227,227))
            
            # Subtract the ImageNet mean
            #img -= imagenet_mean
            
            # Reshape as needed to feed into model
            img = img.reshape((1,227,227,3))
            
            # Run the session and calculate the class probability
            probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
            r_fc7 = sess.run(fc7,feed_dict={x:img ,keep_prob:1.0})
            r_conv1 = sess.run(conv1,feed_dict={x:img ,keep_prob:1.0})
            r_conv5 = sess.run(conv5,feed_dict={x:img ,keep_prob:1.0})

            # Cria diretório para salvar features        
            os.makedirs(os.path.dirname(dir));

            # Salva features de cada layer
            # CONV1 55x55x96
            file = open(dir+"/conv1.txt", 'w')
            for i, item in enumerate(r_conv1[0]):
                for i2, item2 in enumerate(r_conv1[0][i]):
                    for i3, item3 in enumerate(r_conv1[0][0][i2]):
                        file.write("%s\n" % item3)
            file.close();

            # CONV5 13x13x255
            file = open(dir+"/conv5.txt", 'w')
            for i, item in enumerate(r_conv5[0]):
                for i2, item2 in enumerate(r_conv5[0][i]):
                    for i3, item3 in enumerate(r_conv5[0][0][i2]):
                        file.write("%s\n" % item3)
            file.close();

            # FC7 1x1x4096
            file = open(dir+"/conv7.txt", 'w')
            for item in r_fc7[0]:
                file.write("%s\n" % item)
            file.close();

            
            # Get the class name of the class with the highest probability
            class_name = class_names[np.argmax(probs)]