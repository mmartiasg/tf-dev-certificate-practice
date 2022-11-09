#!/usr/bin/env python
# coding: utf-8

# # Implement linear regression on tensorflow with gradient tape

# In[ ]:


import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# In[ ]:


DATASETS = "datasets/"


# # Read dataset

# In[ ]:


with open(DATASETS+os.sep+"winequality-red.csv", "r") as file:
    raw_data = file.read()


# # Remove columns

# In[ ]:


raw_dataset = raw_data.split("\n")[1:]
columns = raw_data.split("\n")[0].split(",")

N_ROWS = len(raw_dataset)
N_FEATURES = len(columns)


# In[ ]:


columns


# In[ ]:


print("ROWS: ", N_ROWS)
print("Features: ", N_FEATURES)


# In[ ]:


dataset = np.zeros((N_ROWS, N_FEATURES))

i = 0
for row in raw_dataset:
    j = 0
    for feature in row.split(","):
        dataset[i][j] = float(feature)
        j+=1
    i+=1


# In[ ]:


dataset[0].shape


# In[ ]:


dataset[:, 7:]


# # Split dataset into features and target

# In[ ]:


N_SEQ_FEATURES = 10


# In[ ]:


x_train = tf.constant(dataset[:, :N_SEQ_FEATURES], dtype="float32")
y_train = tf.constant(dataset[:, -1], dtype="float32")


# # Split dataset into train and validation
# 
# into 70/30
# 
# - Shuffle dataset
# - Select N for Train and N for Validation base on the proportio 70/20/10

# In[ ]:


shuffled_dataset = np.random.shuffle(dataset)
N_train = int(dataset.shape[0]*0.7)+1
N_val = int(dataset.shape[0]*0.2)+1
N_test = dataset.shape[0]-N_train-N_val


# In[ ]:


print("Train rows: ", N_train)
print("Validation rows: ", N_val)
print("Test rows: ", N_test)


# # Split data 
# Into 
# - Train
# - Val
# - Test 
# 
# And X for features and y for the target

# In[ ]:


train_X = dataset[:N_train, :N_SEQ_FEATURES]
train_y = dataset[:N_train, -1]

val_X = dataset[:N_val, :N_SEQ_FEATURES]
val_y = dataset[:N_val, -1]

test_X = dataset[:N_test, :N_SEQ_FEATURES]
test_y = dataset[:N_test, -1]


# # First model Baseline AVG 

# In[ ]:


baseline_prediction = train_y.mean()

print("Pred for baseline: ", baseline_prediction)


# # Metric

# In[ ]:


((val_y - baseline_prediction)**2).sum()/val_y.shape[0]


# # Scale data

# In[ ]:


np.array([4,4,4,4])/np.array([2, 2, 2, 2])


# In[ ]:


train_X.max(axis=0).shape


# In[ ]:


train_X.max(axis=0)


# In[ ]:


train_X.min(axis=0)


# In[ ]:


train_max = train_X.max(axis=0)
train_min = train_X.min(axis=0)

Q_factor = 100

train_X -= train_min
train_X /= (train_max-train_min) * Q_factor

val_X -= train_min
val_X /= (train_max-train_min) * Q_factor

test_X -= train_min
test_X /= (train_max-train_min) * Q_factor


# In[ ]:


# train_X[train_X>1]


# In[ ]:


# train_X[train_X<0].min()


# In[ ]:


# for i in range(train_X.shape[0]):
#     for j in range(train_X.shape[1]):
#         if train_X[i][j] == -4828.0538789457505:
#             print(train_X[i])


# In[ ]:


# assert np.sum(train_X>1)==0
# assert np.sum(train_X<0)==0


# # First model Regresion
# 
# Y = W.X + b

# In[ ]:


W_N_DIMS = train_X.shape[1]
B_N_DIMS = train_X.shape[0]


# In[ ]:


W = tf.Variable(np.random.ranf((1, W_N_DIMS)), dtype='float32')
b = tf.Variable(np.random.ranf((B_N_DIMS, 1)), dtype='float32')


# In[ ]:


W


# In[ ]:


b


# # Inputs

# In[ ]:


train_x_tensor = tf.constant(train_X, dtype='float32')
train_y_tensor = tf.constant(train_y, dtype='float32')


# W (1, DIM) * X (N, DIM)

# # Try random values first

# In[ ]:


y = tf.matmul(W, tf.transpose(train_x_tensor)) + b


# In[ ]:


tf.math.reduce_mean(tf.pow(tf.subtract(y, train_y_tensor), 2))


# # Try with gradien tape to fix the weights
# 
# Problems I ran into
# 
# - I got none because I have declared W and b as constants!!! derivative is 0!
# - Im getting nan an inf!!?
# - Mac tensorflow cant work with float16!

# In[ ]:


with tf.GradientTape() as tape:
    y = tf.matmul(train_x_tensor, tf.transpose(W)) + b
    loss = tf.math.reduce_mean(tf.pow(tf.subtract(y, train_y_tensor), 2))

gradient_loss_w_b = tape.gradient(loss, [W, b])


# W [1, 11]
# train_x_tensor[1120, 11]
# 
# W*train_x_tensor [1120, 1]
# 
# b [1120, 1]

# In[ ]:


tf.matmul(train_x_tensor, tf.transpose(W))+b


# In[ ]:


y


# In[ ]:


print("w")
print(W)
print("b")
print(b)
print("W gradient")
print(gradient_loss_w_b[0])
print("b gradient")
print(gradient_loss_w_b[1])
print("updated W")
print(W.assign_sub(tf.constant(0.01, dtype="float32")*gradient_loss_w_b[0]))
print("updated b")
print(b.assign_sub(tf.constant(0.01, dtype="float32")*gradient_loss_w_b[1]))


# # Training loop

# In[ ]:


train_x_tensor[0]


# In[ ]:


W[0]


# In[ ]:


b[0]


# In[ ]:


#init params
W = tf.Variable(np.random.ranf((1, W_N_DIMS)), dtype='float32')
b = tf.Variable(np.random.ranf((B_N_DIMS, 1)), dtype='float32')

#init epsilon
epsilon = tf.constant(0.01, dtype='float32')

for epoc in tqdm(range(5000)):
    # Feed-forward pass
    with tf.GradientTape() as tape:
        y = tf.matmul(train_x_tensor, tf.transpose(W)) + b
        loss = tf.reduce_mean(tf.square(train_y_tensor-y))

    if epoc%1000==0:
        print("Loss")
        print(loss)
        print("random value y pred: ")
        print(y[0])
        print("random value y: ")
        print(train_y_tensor[0])

    #backward - pass
    w_grad, b_grad = tape.gradient(loss, [W, b])

    W.assign_sub(epsilon*w_grad)
    b.assign_sub(epsilon*b_grad)


# # Batch approach is faster to converge
# 
# as it is able to adjust weights faster

# In[ ]:


#init params
BATCH_SIZE = 32
W = tf.Variable(np.random.ranf((1, W_N_DIMS)), dtype='float32')
b = tf.Variable(np.random.ranf((BATCH_SIZE, 1)), dtype='float32')

#init epsilon
epsilon = tf.constant(0.01, dtype='float32')

for epoc in tqdm(range(50000)):
    
    # Feed-forward pass
    for batch in range(0, B_N_DIMS, BATCH_SIZE):
        with tf.GradientTape() as tape:
            # print(f"Range: {batch} , {batch+BATCH_SIZE}")
            y = tf.matmul(train_x_tensor[batch:batch+BATCH_SIZE], tf.transpose(W)) + b
            loss = tf.reduce_mean(tf.square(train_y_tensor-y))
        #backward - pass
        w_grad, b_grad = tape.gradient(loss, [W, b])

        W.assign_sub(epsilon*w_grad)
        b.assign_sub(epsilon*b_grad)

    if epoc%1000==0:
        print("Loss")
        print(loss)
        print("random value y pred: ")
        print(y[0])
        print("random value y: ")
        print(train_y_tensor[0])


# # With RMS to avoid getting stuck in a local minima around [0.6165769, 0.6733488]

# In[ ]:


#init params
W = tf.Variable(np.random.ranf((1, W_N_DIMS)), dtype='float32')
b = tf.Variable(np.random.ranf((B_N_DIMS, 1)), dtype='float32')

velocity_w = tf.Variable(np.zeros((1, W_N_DIMS)), dtype='float32')
velocity_b = tf.Variable(np.zeros((B_N_DIMS, 1)), dtype='float32')
past_velocity_w = tf.Variable(np.zeros((1, W_N_DIMS)), dtype='float32')
past_velocity_b = tf.Variable(np.zeros((B_N_DIMS, 1)), dtype='float32')
momentum = tf.constant(0.1, dtype='float32')

#init epsilon
epsilon = tf.constant(0.01, dtype='float32')

for epoc in tqdm(range(5000)):
    # Feed-forward pass
    with tf.GradientTape() as tape:
        y = tf.matmul(train_x_tensor, tf.transpose(W)) + b
        loss = tf.reduce_mean(tf.square(train_y_tensor-y))

    if epoc%1000==0:
        print("Loss")
        print(loss)
        print("random value y pred: ")
        print(y[0])
        print("random value y: ")
        print(train_y_tensor[0])

    #backward - pass
    w_grad, b_grad = tape.gradient(loss, [W, b])

    velocity_w.assign(past_velocity_w*momentum-epsilon*w_grad)
    velocity_b.assign(past_velocity_b*momentum*epsilon*b_grad)

    W.assign_add(velocity_w*momentum-epsilon*w_grad)
    b.assign_add(velocity_b*momentum-epsilon*b_grad)

    past_velocity_w.assign(velocity_w)
    past_velocity_b.assign(velocity_b)


# # Batch with momemtum
# 
# # With RMS to avoid getting stuck in a local minima around [0.6165769, 0.6733488]

# In[ ]:


#init params
BATCH_SIZE = 32
W = tf.Variable(np.random.ranf((1, W_N_DIMS)), dtype='float32')
b = tf.Variable(np.random.ranf((BATCH_SIZE, 1)), dtype='float32')

#RMS
velocity_w = tf.Variable(np.zeros((1, W_N_DIMS)), dtype='float32')
velocity_b = tf.Variable(np.zeros((BATCH_SIZE, 1)), dtype='float32')
past_velocity_w = tf.Variable(np.zeros((1, W_N_DIMS)), dtype='float32')
past_velocity_b = tf.Variable(np.zeros((BATCH_SIZE, 1)), dtype='float32')
momentum = tf.constant(0.1, dtype='float32')

#init epsilon
epsilon = tf.constant(0.01, dtype='float32')

for epoc in tqdm(range(50000)):
    
    # Feed-forward pass
    for batch in range(0, B_N_DIMS, BATCH_SIZE):
        with tf.GradientTape() as tape:
            # print(f"Range: {batch} , {batch+BATCH_SIZE}")
            y = tf.matmul(train_x_tensor[batch:batch+BATCH_SIZE], tf.transpose(W)) + b
            loss = tf.reduce_mean(tf.square(train_y_tensor-y))
 
        #backward - pass
        w_grad, b_grad = tape.gradient(loss, [W, b])

        velocity_w.assign(past_velocity_w*momentum-epsilon*w_grad)
        velocity_b.assign(past_velocity_b*momentum*epsilon*b_grad)

        W.assign_add(velocity_w*momentum-epsilon*w_grad)
        b.assign_add(velocity_b*momentum-epsilon*b_grad)

        past_velocity_w.assign(velocity_w)
        past_velocity_b.assign(velocity_b)

    if epoc%1000==0:
        print("Loss: ", loss)
        print(f"First sample training set prediction {y[0]} - real value {train_y_tensor[0]}")

