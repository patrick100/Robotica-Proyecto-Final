# carnd-traffic-sign-classifier-project/Traffic_Sign_Classifier-Copy1.ipynb
# Load pickled data
import pickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import skimage.morphology as morp
import csv
import os
import cv2

import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix

import threading
import socket  




def Gray(ImgSet):
    Num_ImgSet = len(ImgSet)
    shape = ImgSet[0].shape
    NewSet = []
    #if Num_ImgSet > 1:
    # pixel128 = np.ones_like(ImgSet[0]) * 128
    NewImg = np.zeros((shape[0], shape[1], 1))

    for i in range(Num_ImgSet):
        img = ImgSet[i]
        tmp = np.array([np.dot(img[..., :3], [0.299, 0.587, 0.114])])
        NewImg = np.rollaxis(tmp, 0, 3) #(matrix, 需调整的轴, 目标位置)
        NewSet.append(NewImg)

    #else:
    #    print('Input should be dataset instead of a single image')
    return NewSet




def Sign(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    ConvStrides = [1, 1]
    PoolStrides = [2, 2]
    L1Filter = [5, 5, 1] # [Filter height, Filter width, Input depth]
    L1Output = [(32-L1Filter[0]+1) / ConvStrides[0], (32-L1Filter[1]+1) / ConvStrides[1], 6 ]  # VALID Padding output computation formula
    L2Filter = [3, 3, L1Output[2]]
    L2Output = [(L1Output[0]/PoolStrides[0] - L2Filter[0] + 1)/ConvStrides[0],
                (L1Output[1]/PoolStrides[1] - L2Filter[1] + 1)/ConvStrides[1], 20]
    L3Filter = [3, 3, L2Output[2]]
    L3Output = [(L2Output[0]/PoolStrides[0] - L3Filter[0] + 1)/ConvStrides[0],
                (L2Output[1]/PoolStrides[1] - L3Filter[1] + 1)/ConvStrides[1], 60]
    L4Input = int((L3Output[0]/PoolStrides[0]) * (L3Output[1]/PoolStrides[1]) * L3Output[2])
    L4Output = 160
    L5Output = 80


    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(L1Filter[0], L1Filter[1], L1Filter[2], L1Output[2]), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(L1Output[2]))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x16x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x16x6 Output = 12x12x20.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(L2Filter[0], L2Filter[1], L2Filter[2], L2Output[2]), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(L2Output[2]))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 12x12x20. Output = 6x6x20.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 3: Convolutional. Input = 6x6x20. Output = 4x4x60.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(L3Filter[0], L3Filter[1], L3Filter[2], L3Output[2]), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(L3Output[2]))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Pooling. Input = 4x4x60. Output = 2x2x60.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Flatten. Input = 2x2x60. Output = 240.
    fc0 = flatten(conv3)

    # Layer 4: Fully Connected. Input = 240. Output = 160.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(L4Input, L4Output), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(L4Output))
    fc0 = tf.nn.dropout(fc0, keep_prob)
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 5: Fully Connected. Input = 160. Output = 80.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(L4Output, L5Output), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(L5Output))
    # fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 6: Fully Connected. Input = 80. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(L5Output, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    # fc2 = tf.nn.dropout(fc2, keep_prob)
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits



def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples





def buscar_imagen(id_labels,myids):
	
	for i in range(len(id_labels)):
		for j in range(len(myids)):
			if(id_labels[i]==myids[j]):
				return myids[j]
    




def y_predict_model(Input_data, top_k=5):

    num_examples = len(Input_data)
    y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
    y_prob = np.zeros((num_examples, top_k))
    with tf.Session() as sess:

        saver.restore(sess, './Traffic-Sign-CNNmodel')
        #y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(VGGNet_Model.logits), k=top_k), feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})
        y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=top_k), feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})


    return y_prob, y_pred



signs = []
with open('./signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()


n_classes = 43

EPOCHS = 20
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

'''Training Pipeline'''
rate = 0.001

logits = Sign(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

'''Model Evaluation'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()





keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers
saver = tf.train.import_meta_graph("./Traffic-Sign-CNNmodel.meta")


def enviar_instruccion(mensaje):

    # Create a socket object 
    s = socket.socket()          
    # Define the port on which you want to connect 
    port = 12345
    #ponemos la direccion del robot                
    s.connect(("192.168.43.177", port))
    #s.connect(("192.168.0.10", port))
    #print (s.recv(1024).decode())
    mymensaje = mensaje 
    s.sendall(mensaje.encode())
    s.close()  


def cargar_instruccion():
    
    # Loading and resizing new test images# Loadin 
    new_test_images = []
    path = './test/'
    for image in os.listdir(path):
        img = cv2.imread(path + image)
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_test_images.append(img)
    new_IDs = [13, 3, 14, 27, 17]

    print("Number of new testing examples: ", len(new_test_images))

    new_test_images_preprocessed = Gray(np.asarray(new_test_images))
    y_prob, y_pred = y_predict_model(new_test_images_preprocessed)


    myids = [14,33,34]
    for i in range(len(new_test_images_preprocessed)):
        print("imagen "+ str(i) +" "+ str(signs[y_pred[i][0]]+" pos "+str(y_pred[i][0]) ) )
        #print("probabilidades: ")
        #print(np.arange(1, 6, 1), y_prob[i, :])
        #print(y_pred[i])

        pos =buscar_imagen(y_pred[i],myids)

        print("THE REAL POS "+str(pos))

        enviar_instruccion(str(pos))
        #enviar_instruccion(str(pos))



        #labels = [signs[j] for j in y_pred[i]]
        #print(labels)    

#def main():
    #hilo = threading.Thread(target=myserver)
    #hilo.start()
    #hilo.join() 


#main()


def myserver():
 
    # next create a socket object 
    s = socket.socket()          
    print("Socket successfully created")
      
    # reserve a port on your computer in our 
    # case it is 12345 but it can be anything 
    port = 1234                
    #s.bind(("192.168.0.10", port))         
    s.bind(("", port))
    print("socket binded to %s" %(port)) 
    # put the socket into listening mode 
    s.listen(5)      
    print("socket is listening")            
      
    # a forever loop until we interrupt it or  
    # an error occurs 
    while True: 
      
        # Establish connection with client. 
        c, addr = s.accept() 

        #data = c.recv(1024).decode()
        #if not data: break
        #print("Recieved: "+(data))

        print("PROCESANDO Y ENVIANDO INSTRUCCION")
        cargar_instruccion()
        #enviar_instruccion("GIRARRRR DER")

        #print 'Got connection from', addr 

        # send a thank you message to the client.  
        #c.send('Thank you for connecting') 

        # Close the connection with the client 
        c.close() 

def inicializar_server():
    hilo = threading.Thread(target=myserver)
    hilo.start()
    hilo.join()
    myserver() 


inicializar_server()
#cargar_instruccion()