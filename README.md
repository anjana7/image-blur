# image-blur

I have used the keras resnet model for feature extraction. This gives me 1000 features for each image. Thus my feature vector consist of 995 vectors (995 images) each of 1000 features. I am passing this to the    model using the scikit learn package. After the model is trained I have the weights saved.
I am then parsing through the evaluation images and passing each image to the model to test the output. The results of the test images is saved in an xlsx file so I use the pandas dataframe model to read the data. I store each of the classes in a classify array. I then compare the result recieved and the actual result to calculate the accuracy of the predicted model. 

We require Keras and Sklearn naivebayes packages for the model to run

pip install numpy, scipy, scikit-learn

running it on jupyter will help in importing needed softwares too
