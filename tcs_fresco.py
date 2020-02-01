from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np

#2

# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainy), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainy = to_categorical(trainy)
    testY = to_categorical(testY)
    return trainX, trainy, testX, testY


#3

seed=9

from sklearn.model_selection import StratifiedShuffleSplit

data_split = StratifiedShuffleSplit(test_size=0.08,random_state=seed )
for train_index, test_index in data_split.split(trainX, trainy):

    split_data_92, split_data_8 = trainX[train_index], trainX[test_index]

    split_label_92, split_label_8 = trainy[train_index], trainy[test_index]
train_test_split = StratifiedShuffleSplit( test_size=0.3,random_state=seed)
#test_size=0.3 denotes that 30 % of the dataset is used for testing.


#4


for train_index, test_index in train_test_split.split(split_data_8,split_label_8):

    train_data_70, test_data_30 = split_data_8[train_index], split_data_8[test_index]

    train_label_70, test_label_30 = split_label_8[train_index], split_label_8[test_index]
train_data = train_data_70 #assigning to variable train_data

train_labels = train_label_70 #assigning to variable train_labels

test_data = test_data_30

test_labels = test_label_30
print('train_data : ',    train_data.shape                )

print('train_labels : ',    train_labels.shape             )

print('test_data : ',       test_data.shape            )

print('test_labels : ',      test_labels.shape          )


#5

# definition of normalization function

def normalize(data, eps=1e-8):

    data -=data.mean(axis=(0,1,2),keepdims=True)

    std =np.sqrt(data.var(axis=(0,1,2),ddof=1,keepdims=True))

    std[std < eps] = 1.

    data /= std

    return data
train_data=train_data.astype('float64')
test_data=test_data.astype('float64')
# calling the function

train_data = normalize(train_data)

test_data = normalize(test_data)
# prints the shape of train data and test data

print('train_data: ',   train_data.shape        )

print('test_data: ',   test_data.shape        )


#6

# Computing whitening matrix

train_data_flat = train_data.reshape(train_data.shape[0], -1).T

test_data_flat = test_data.reshape(test_data.shape[0], -1).T

print('train_data_flat: ',   train_data_flat.shape       )

print('test_data_flat: ',         test_data_flat.shape      )



train_data_flat_t = train_data_flat.T

test_data_flat_t = test_data_flat.T


#7


from sklearn.decomposition import PCA

# n_components specify the no.of components to keep

train_data_pca =PCA(n_components=train_data_flat.shape[0]).fit_transform(train_data_flat)

test_data_pca =PCA(n_components=test_data_flat.shape[0]).fit_transform(test_data_flat)
#
print(     'pca train', train_data_pca.shape           )

print(    'pca test',   test_data_pca.shape       )

train_data_pca = train_data_pca.T

test_data_pca = test_data_pca.T

#8

# from skimage import color
# def svdFeatures(input_data):
#
#     svdArray_input_data=[]
#
#     size = input_data.shape[0]
#
#     for i in range (0,size):
#
#         img=color.rgb2gray(input_data[i])
#
#         U, s, V = np.linalg.svd(img, full_matrices=False);
#
#         S=[s[i] for i in range(28)]
#
#         svdArray_input_data.append(S)
#
#         svdMatrix_input_data=np.matrix(svdArray_input_data)
#
#     return svdMatrix_input_data
# svdMatrix_input_data
#
#
# # apply SVD for train and test data
#
# train_data_svd=svdFeatures(train_data)
#
# test_data_svd=svdFeatures(test_data)
# print(train_data_svd.shape)
# print(test_data_svd.shape)
#
#
# #9
#
# from sklearn import svm #Creating a svm classifier model
#
# clf = svm.SVC(  gamma=0.001   , probability=True                 ) #train_data_flat_tModel training
#
# train = clf.fit(train_data_flat_t,train_labels)
# predicted=clf.predict(test_data_flat_t)
#
# score = clf.score(test_data_flat_t,test_labels)
# print("score",score)
#
# with open('output.txt', 'w') as file:
#     file.write(str(np.mean(score)))
#




