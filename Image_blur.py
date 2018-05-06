from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import glob
from sklearn.naive_bayes import GaussianNB
import pandas

model = ResNet50(weights='imagenet')

def features(img):
    img = image.load_img(img, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    resnet_feature = model.predict(img_data)
    return(np.array(resnet_feature).flatten())

def testing():
    patho = 'path to evaluation set'
    pathi = 'path to evaluation set 2'
    list_in = ['NaturalBlurSet/','DigitalBlurSet/']
    list_op = ['NaturalBlurSet.xlsx','DigitalBlurSet.xlsx']
    
    classify = []
    
    clas1 = (pandas.read_excel(patho+list_op[0]))
    df = pandas.DataFrame(clas1)
    for i in range(1000):
        y = (df.iat[i,1])
        classify.append(y)
    
    
    clas2 = (pandas.read_excel(patho+list_op[1]))
    df = pandas.DataFrame(clas2)
    for i in range(480):
        y = (df.iat[i,1])
        classify.append(y)
    
    Y = np.array(classify)
    Y = Y.reshape(Y.shape[0],1)
    
    TP,FP,TN,FN = 0,0,0,0
    j = 0
    for i in range(len(list_in)):
        for img in glob.glob(pathi+list_in[i]+"*.jpg"):
            f = features(img)
            f = f.reshape(1,f.shape[0])
            res = clf.predict(f)
            print(res)
            if res == -1:
                if Y[j] == -1:
                    TN+=1
                else:
                    FN+=1
            else:
                if Y[j] == 1:
                    TP+=1
                else:
                    FP+=1
            j+=1
            
    accuracy = ((TP+TN+1)/(TP+FP+TN+FN+1))
    print(accuracy)
    print(TP,TN,FP,FN)
    return('Success')    


list_in = ['Naturally-Blurred/','Artificially-Blurred/','Undistorted/']
path = 'path to training set'

feature = []
output = []
for i in range(len(list_in)):
    for img in glob.glob(path+list_in[i]+"*.jpg"):
        f = features(img)
        feature.append(f)
        if (i != 2):
            output.append(-1)
        else:
            output.append(1)
            
for i in range(len(list_in)):
    for img in glob.glob(path+list_in[i]+"*.JPG"):
        f = features(img)
        feature.append(f)
        if (i != 2):
            output.append(-1)
        else:
            output.append(1)
            
print(len(feature))
print(len(feature[0]))

X = np.array(feature)
print(X.shape)

clf = clf = GaussianNB()
clf = clf.fit(X, output)
testing()
