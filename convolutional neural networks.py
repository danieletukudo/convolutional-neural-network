import tensorflow
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D,Input,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D



def funtionmodel():
 
   my_input = Input(shape = (28,28,1))
   x =  Conv2D(32,(3,3),activation = 'relu')(my_input)
   x =  Conv2D(64,(3,3),activation = 'relu')(x)
   x = MaxPool2D()(x)
   x = BatchNormalization()(x)

   x = Conv2D(128,(3,3),activation = 'relu')(x)
   x = MaxPool2D()(x)
   x = BatchNormalization()(x)

   x = GlobalAveragePooling2D()(x)
   x = Dense(64,activation ='relu')(x)
   x = Dense(10,activation ='softmax')(x)
   model = tensorflow.keras.Model( inputs = my_input,outputs = x)
   return model

class my_costommodel(tensorflow.keras.Model):
  def __init__(self):
    super().__init__()
    self.conv1 =  Conv2D(32,(3,3),activation = 'relu')
    self.conv2  = Conv2D(64,(3,3),activation = 'relu')
    self.maxpool1= MaxPool2D()
    self.BatchNorm1= BatchNormalization()

    self.conv3 =  Conv2D(128,(3,3),activation = 'relu')
    self.maxpool12 = MaxPool2D()
    self.BatchNorm2 = BatchNormalization()

    self.Global =  GlobalAveragePooling2D()
    self.Dense1= Dense(64,activation ='relu')
    self.Dense2 = Dense(10,activation ='softmax')
     
    
    
  def call(self,myinput):
    x = self.conv1(myinput)
    x = self.conv2(x)
    x = self.maxpool1(x)
    x =self.BatchNorm1(x)
    x= self.conv3(x)
    x= self.maxpool12(x)
    x=  self.BatchNorm2(x)
    x= self.Global(x)
    x= self.Dense1(x)
    x= self.Dense2(x)
    return x
    


def display_images(example,labels):
   plt.figure(figsize=(10,10))
   for i in range(25):
      idx = np.random.randint(0,example.shape[0]-1)
      img = example[idx]
      label = labels[idx]
      plt.subplot(5,5,i+1)
      plt.title(str(label))
      plt.tight_layout()
      plt.imshow(img,cmap='gray')
   plt.show()

if __name__=='__main__':
   ( x_train,y_train),(x_test,y_test)  = tensorflow.keras.datasets.mnist.load_data()
   print(x_test.shape)
   print(y_test.shape)
   print(x_train.shape)


   x_train = x_train.astype('float32')/255
   x_test = x_test.astype('float32')/255

   x_train = np.expand_dims(x_train,axis = -1)
   x_test= np.expand_dims(x_train,axis = 1)
   #model =funtionmodel()
   model = my_costommodel()
   model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics = 'accuracy')
   model.fit(x_train,y_train,batch_size = 64,epochs = 3)



  
