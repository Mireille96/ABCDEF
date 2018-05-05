import tensorflow as tf 
import glob
import os
import cv2
import numpy as np

def Read_Train_Images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/DataSet 400x400/*.jpg') 
    grayimage_stack = np.array( [np.array(cv2.imread(imagePath[i],cv2.IMREAD_GRAYSCALE)) for i in range(len(imagePath))] )
    print("Grey Train Images")
    return grayimage_stack

def ReadLabels_Train_Images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/DataSet 388x388/*.jpg') 
    RGBimage_stack = np.array( [np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))] )
    print("RGB Train Images")
    return RGBimage_stack


def Read_GreyTest_Images_in_folder():
    imagePath = glob.glob('/home/meraymedhat/Test/*.jpg')
    grayTestimage_stack = np.array( [np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))] )
    print("Grey Test Images")
    return grayTestimage_stack


def resize_image(image_stack,Height,Width): 
    im_resized_stack = np.array( [np.array(cv2.resize(img, (Height, Width), interpolation=cv2.INTER_CUBIC)) for img in image_stack]) 
    print("Resize Images")
    return im_resized_stack

def weight_variable(shape):
    initial = tf.truncated_normal(shape , stddev =0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#here there is no padding
def FirstThreeLayersconv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1] ,padding='VALID')

def SecondThreeLayersconv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1] , padding ='SAME')

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)   #law condition true haikmal law la2 haitla3 assertion error.
    p = np.random.permutation(len(a))
    return a[p], b[p]

keep_prob = tf.placeholder(tf.float32)
GreyResizeWidth = 400
GreyResizeHeight = 400

LabelResizeWidth = 388
LabelResizeHeight = 388

x = tf.placeholder(tf.float32,shape =[None,GreyResizeWidth*GreyResizeHeight])#400*400
y_ = tf.placeholder(tf.float32,shape=[None,LabelResizeWidth,LabelResizeHeight,3])#388*388*3 3alshan RGB


def convolutional_neural_network(x):
    #aksa 7aga fi el alwan 255 fa 3amlna - 128 bi2olk en aksa lon 3andk 128 then /128 ya3ni b2a range el alwan -1 to 1 (Normalization)
    x_image = (tf.reshape(x,shape = [-1,GreyResizeWidth,GreyResizeHeight,1])-128)/128
   
    #Arc
    #Conv1
    W_con1 = weight_variable([5,5,1,8])
    b_conv1 = bias_variable([8])

    h_conv1 = tf.nn.relu(FirstThreeLayersconv2d(x_image,W_con1) + b_conv1)

    #output here 396*396*8
    #Conv2
    W_con2 = weight_variable([5,5,8,16])
    b_conv2 = bias_variable([16])

    h_conv2 = tf.nn.relu(FirstThreeLayersconv2d(h_conv1,W_con2) + b_conv2)

    #output = 392*392*16

    #Conv3
    W_con3 = weight_variable([5,5,16,32])
    b_conv3 = bias_variable([32])

    h_conv3 = tf.nn.relu(FirstThreeLayersconv2d(h_conv2,W_con3) + b_conv3)
    #output = 388*388*32 

    #Conv4
    W_con4 = weight_variable([5,5,32,64])
    b_conv4 = bias_variable([64])

    h_conv4 = tf.nn.relu(SecondThreeLayersconv2d(h_conv3,W_con4) + b_conv4)
    #output 388*388*64

    #Conv5
    W_con5 = weight_variable([5,5,64,128])
    b_conv5 = bias_variable([128])

    h_conv5 = tf.nn.relu(SecondThreeLayersconv2d(h_conv4,W_con5) + b_conv5)
    #output 388*388*128

    #Conv6
    W_con6 = weight_variable([5,5,128,256])
    b_conv6 = bias_variable([256])

    h_conv6 = tf.nn.relu(SecondThreeLayersconv2d(h_conv5,W_con6) + b_conv6)
    #output 388*388*256

    #Conv7
    W_con7 = weight_variable([5,5,256,3])
    b_conv7 = bias_variable([3])

    h_conv7 = SecondThreeLayersconv2d(h_conv6,W_con7) + b_conv7
    #output 388*388*3

    y_conv = (tf.nn.sigmoid(h_conv7))*255
    #y_conv = tf.image.resize_nearest_neighbor(y_conv,[LabelResizeWidth,LabelResizeHeight])
    return y_conv


GrayImages = Read_Train_Images_in_folder() 
RGBImages_LabelsResized = ReadLabels_Train_Images_in_folder()

GrayImagesResized = GrayImages.reshape([-1,GreyResizeWidth*GreyResizeHeight])

#shuffle 3alshan yat3lam el weights mn kaza category.
unison_shuffled_copies(RGBImages_LabelsResized,GrayImagesResized)

prediction = convolutional_neural_network(x)
#momkan el line da yatla3 result a7san
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.subtract(prediction,y_) ** 2) ** 0.5)


#optimizier
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

def TrainCnn():
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "M:/Automatic Colorization/WeightsModel/model/model.ckpt")
        for epoch in range(15000):
            epoch_loss = 0
            #hatlaf 310 iteration kol batch fiha 20 sora
            for i in range(int(6208/20)):
                print("Batch Num ",i + 1)
                a, c = sess.run([train_step,cross_entropy],feed_dict={x: GrayImagesResized[i*20:(i+1)*20], y_: RGBImages_LabelsResized[i*20:(i+1)*20], keep_prob: 0.5})
                epoch_loss +=c
                save_path = saver.save(sess, "/home/meraymedhat/WeightsModel/model/model.ckpt")
            print("epoch: ",epoch + 1, ",Loss: ",epoch_loss)
          
def testNN():
    saver = tf.train.Saver()
    TempTestGray = Read_GreyTest_Images_in_folder().reshape([-1,GreyResizeWidth*GreyResizeHeight])
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, "/home/meraymedhat/WeightsModel/model/model/model.ckpt") 
        Output = sess.run(prediction,feed_dict = {x:TempTestGray,keep_prob: 0.5})
        #here el loop b 3adad el sawer eli fi folder el test
        for i in range (2086):
          Image = TempTestGray[i]
          Result = np.floor(Output[i])
          cv2.imwrite('/home/meraymedhat/WeightsModel/Results/res.jpg',Result);
          cv2.imwrite('/home/meraymedhat/WeightsModel/Results/input.jpg',Image.reshape([-1,GreyResizeWidth,GreyResizeHeight])[0])
          img = cv2.imread("/home/meraymedhat/WeightsModel/Results/res.jpg")
          cv2.startWindowThread()
          cv2.namedWindow("Colored Image")
          cv2.imshow("Colored Image", img)
          cv2.waitKey(0)                  

TrainCnn()
testNN()