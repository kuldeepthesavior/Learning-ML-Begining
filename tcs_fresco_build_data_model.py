#1

###Sart code
import pandas as pd         #import pandas
data=pd.read_csv('blobs1.csv')
df =  pd.DataFrame(data) #create a dataframe named 'data' from 'blobs.csv' file
###End code
print(df.head())

#2

###Start code
X = df[['feature1', 'feature2']].values
y = df['target'].values                                    #extract target values
###End code
assert X.shape == (5000, 2)
assert y.shape == (5000, )

#3

import matplotlib.pyplot as plt
import matplotlib.colors
colors=['blue','green']
cmap = matplotlib.colors.ListedColormap(colors)
#Plot the figure
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X[:,0], X[:,1], c=y,
           marker= 'o', s=50,cmap=cmap,alpha = 0.5 )
plt.show()

#4

###Start code
X_data = X.T
y_data = y.reshape(1,len(y))
###End code
# print(X_data.shape , y_data.shape)
assert X_data.shape == (2, 5000)
assert y_data.shape == (1, 5000)

#5

layer_dims = [2,25,25,1]

#6

import  tensorflow as tf   #import tensorflow

#7

def placeholders(num_features, num_classes):
    A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))
    Y = tf.placeholder(dtype = tf.float64, shape = ([num_classes,None]))
    return A_0,Y

#8

def initialize_parameters_deep(layer_dims):
    tf.set_random_seed(1)
    L = len(layer_dims)
    parameters = {}
    for l in range(1,L):
        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]], dtype = tf.float64,
                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable("b"+ str(l), shape = [layer_dims[l], 1], dtype= tf.float64, initializer= tf.zeros_initializer() )
    return parameters
#     print(parameters)
#

#9

def linear_forward_prop(A_prev,W,b, activation):
    ###Start code
    Z = tf.add(tf.matmul(W,A_prev),b)
    if activation == "sigmoid":
        A = Z
    elif activation == "relu":
        A =tf.nn.relu(Z)
    ###End code
    return A

#10

def l_layer_forwardProp(A_0, parameters, drop_out = False):
    A = A_0
    L = len(parameters)//2                               #number of layers
    for l in range(1,L):
        A_prev = A
        ###Start code
        A =  linear_forward_prop(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu" )  #call linear_forward_prop to return the output from current layer
        ###End code
        if drop_out:                                    #check if dropout == True, if true apply dropout to current layer's output.
            A = tf.nn.dropout(x=A,keep_prob=0.8)  # call tensoflow's droupout function to deactivate output A, set keep_prob = 0.8
    A = linear_forward_prop(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid" )   # return output from final layer.
    return A

#11

def final_cost(Z_final, Y, parameters, regularization=False, lambd=0):
    # define original cost
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final, labels=Y)
    if regularization:
        # initialize regularization term to zero
        reg_term = 0
        L = len(parameters) // 2
        for l in range(1, L +1):
            # use tensorflow's l2 regularization to calculate regularization term for each later and
            # sum it up to previous layer's regularization term

            ###Start code
            reg_term += tf.nn.l2_loss(parameters['W' + str(l)])

        # multiply lambd/2 to reg_term to add it to original cost
        cost += tf.reduce_mean(reg_term*(lambd/2))
        ##End code
    return tf.reduce_mean(cost)

#12

def deep_net(X_train, Y_train, layer_dims, learning_rate, num_iter, regularization=False, lambd=0, drop_out=False):
    tf.reset_default_graph()
    num_features = layer_dims[0]
    ###Start code
    A_0, Y = placeholders(num_features)  # call placeholder function to initialize placeholders A_0 and Y
    parameters = initialize_parameters_deep(
        layer_dims)  # Initialse Weights and bias using initialize_parameters_deep() with layer_dims as parameters
    Z_final = l_layer_forwardProp(A_0, parameters,
                                  drop_out)  # call the function l_layer_forwardProp() to define the final output

    # call final_cost() function to return the cost that has to be minimized during gradient descent
    cost = final_cost(Z_final, Y, parameters, regularization, lambd)

    # call tensorflow's gradient descent optimizer function with minimize cost
    train_net = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # End code
    init = tf.global_variables_initializer()
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iter):
            _, c = sess.run([train_net, cost], feed_dict={A_0: X_train, Y: Y_train})
            if i % 100 == 0:
                costs.append(c)
            if i % 1000 == 0:
                print(c)
        with open('output2.txt', 'w') as file:
            file.write("cost = %f " % costs[-1])
        plt.ylim(min(costs) + 0.1, max(costs), 4, 0.01)
        plt.xlabel("epoches per 100")
        plt.ylabel("cost")
        plt.plot(costs)
        plt.show()
        params = sess.run(parameters)
    return params

#13

def predict(A_0, parameters):
    with tf.Session() as sess:
        Z = l_layer_forwardProp(A_0, parameters, drop_out= False)
        A = sess.run(tf.round(tf.sigmoid(Z)))
    return A

#14

import numpy as np
def plot_decision_boundary1( X, y, model):
    plt.clf()
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    colors=['blue','green']
    cmap = matplotlib.colors.ListedColormap(colors)
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    A = model(np.c_[xx.ravel(), yy.ravel()])
    A = A.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, A, cmap="spring")
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, s=8,cmap=cmap)
    plt.title("Decision Boundary for learning rate:")
    plt.show()


#15

###Start code
parameters =deep_net(X_data,y_data, layer_dims, 0.01, 10000, regularization = False, lambd = 0, drop_out = False)
##End code
plot_decision_boundary1(X_data,y,lambda x: predict(x.T,parameters))

#16

##Start code
parameters = deep_net(X_data,y_data, layer_dims, 0.01, 10000, regularization = True, lambd = 0.02, drop_out = False)
###end code
plot_decision_boundary1(X_data,y,lambda x: predict(x.T,parameters))

#17

#Start code
parameters = deep_net(X_data,y_data, layer_dims, 0.01, 10000, regularization = False, lambd = 0, drop_out = True)
###End code
plot_decision_boundary1(X_data,y,lambda x: predict(x.T,parameters))
