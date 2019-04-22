
# coding: utf-8

# ## Pre-work Lab for Google Cloud ML Practitioner Training (Aug 6-10, 2018)
# 
# In this lab, you will create a custom CNN model for the CIFAR10 data set.
# 
# Helpful references:
# - https://www.youtube.com/watch?reload=9&v=eBbEDRsCmv4
# - https://www.tensorflow.org/guide/custom_estimators
# - https://www.tensorflow.org/guide/datasets_for_estimators

# In[1]:


import tensorflow as tf
import time
import os
import numpy as np


# ### DO: Create two functions that will be reused in your custom model function.
# 
# _conv:
# - Convolution block to include conv, relu, and max pool.
# - Use tf.variable_scope.
# - Input should include feature, kernel (convolution filter), variable scope name
# 
# _dense:
# - Dense block with relu.  Since last layer is logits, make relu switchable on/off.
# - Use tf.variable_scope.
# - Since last layer is logits, make relu switchable on/off.
# - Input should include feature, in/out sizes, variable scope name, relu(true/false)
# 

# In[2]:


# Convolution Block

def _conv(x,kernel,name):
    with tf.variable_scope(name):
        W = tf.get_variable(initializer=tf.truncated_normal(shape=kernel,stddev=0.01),name='W')
        b = tf.get_variable(initializer=tf.constant(0.0,shape=[kernel[3]]),name='b')
        conv = tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')
        activation = tf.nn.relu(tf.add(conv,b))
        pool = tf.nn.max_pool(activation,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return pool

# Dense Block

def _dense(x,size_in,size_out,name,relu=False):
    with tf.variable_scope(name):
        flat = tf.reshape(x,[-1,size_in])
        W = tf.get_variable(initializer=tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
        b = tf.get_variable(initializer=tf.constant(0.0,shape=[size_out]),name='b')
        activation = tf.add(tf.matmul(flat,W),b)
        if relu==True:
            activation = tf.nn.relu(activation)
        return activation


# ### DO: Create custom model function.
# 
# 5 sections of the model function
# 
# #### 1. INFERENCE MODEL
# 
# Recommended architecture:
# - conv1: kernel = 5x5x128
# - conv2: kernel = 5x5x128
# - conv3: kernel = 3x3x256
# - conv4: kernel = 3x3x512
# - dense: hidden_units (tunable hyperparam)
# 
# Use:
# - tf.nn - this will allow for metric collection later.
# 
# #### 2. CALCULATIONS AND METRICS
# 
# Implement:
# - Prediction dictionary {classes, logits, probabilities}.
# - Loss function: Cross Entropy.
# - Accuracy for both training and eval using tf.metrics.
# 
# #### 3. MODE = PREDICT
# 
# Implement:
# - EstimatorSpec for PREDICT.
# 
# #### 4. MODE = TRAIN
# 
# Implement:
# - Optimizer = Stochastic Gradient Descent.
# - EstimatorSpec for TRAIN.
# - Optional: Exponential Decay Learning Rate.
# 
# #### 5. MODE = EVAL
# 
# Implement:
# - EstimatorSpec for EVAL.
# 

# In[3]:


def cnnmodel_fn(features, labels, mode, params):
    
    #### 1 INFERNCE MODEL
    
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    conv1 = _conv(input_layer,kernel=[5,5,3,128],name='conv1')
    conv2 = _conv(conv1,kernel=[5,5,128,128],name='conv2')
    conv3 = _conv(conv2,kernel=[3,3,128,256],name='conv3')
    conv4 = _conv(conv3,kernel=[3,3,256,512],name='conv4')
    dense = _dense(conv4,size_in=2*2*512,size_out=params['dense_units'],
                   name='Dense',relu=True)
    if mode==tf.estimator.ModeKeys.TRAIN:
        dense = tf.nn.dropout(dense,params['drop_out'])
    logits = _dense(dense,size_in=params['dense_units'],
                    size_out=10,name='Output',relu=False)
        
    #### 2 CALCULATIONS AND METRICS
    
    predictions = {"classes": tf.argmax(input=logits,axis=1),
                   "logits": logits,
                   "probabilities": tf.nn.softmax(logits,name='softmax')}
    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    if (mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits,axis=1))
        metrics = {'accuracy':accuracy}
        
    #### 3 MODE = PREDICT
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs)

    #### 4 MODE = TRAIN

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params['learning_rate'],tf.train.get_global_step(),
            decay_steps=100000,decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
    
    #### 5 MODE = EVAL
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=metrics)


# ### DO: Create function to deserialize tfrecords files.
# 
# Implement parse_tfrecord function:
# - Feature input: {idx: tf.int64, label: tf.int64, image: tf.string}
# - Return: image, label
#     

# In[4]:


def parse_tfrecord(example):
    feature={'idx'     : tf.FixedLenFeature((), tf.int64),
             'label'   : tf.FixedLenFeature((), tf.int64),
             'image'   : tf.FixedLenFeature((), tf.string, default_value="")}
    parsed = tf.parse_single_example(example, feature)
    image = tf.decode_raw(parsed['image'],tf.float64)
    image = tf.cast(image,tf.float32)
    image = tf.reshape(image,[32,32,3])
    return image, parsed['label']


# ### DO: Create two helper functions:
# 
# Implement image_scaling function:
# - Image scaling.
# - Applied always
# 
# Implement distort function:
# - Applied only in training.
# - Resize, crop, flip randomly.

# In[5]:


def image_scaling(x):
    return tf.image.per_image_standardization(x)

def distort(x):
    x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
    x = tf.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x


# ### DO: Create input function using tf.data
# 
# Create dataset_input_fn that implements:
# - Use TFRecordDataset to read tfrecord files.
# - Apply parse function you created above.
# - Apply image scaling function you created above.
# - Apply distort function on for training.
# - Suffle only for training.
# - Use prefetch
# - Optional: parallelize threads wherever possible.

# In[6]:


def dataset_input_fn(params):
    dataset = tf.data.TFRecordDataset(
        params['filenames'],num_parallel_reads=params['threads'])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=params['threads'])
    dataset = dataset.map(lambda x,y: (image_scaling(x),y),num_parallel_calls=params['threads'])
    if params['mode']==tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.map(lambda x,y: (distort(x),y),num_parallel_calls=params['threads'])
        dataset = dataset.shuffle(buffer_size=params['shuffle_buff'])
    dataset = dataset.repeat()
    dataset = dataset.batch(params['batch'])
    dataset = dataset.prefetch(8*params['batch'])
    return dataset


# Create dictionary for parameters

# In[7]:


model_params  = {'drop_out'      : 0.2,
                 'dense_units'   : 1024,
                 'learning_rate' : 1e-3
                }


# ### DO: Create RunConfig
# 
# Implement RunConfig:
# - Save checkpoints ever 300 seconds.
# - Keep up to 5 checkpoint history.

# In[8]:


config = tf.estimator.RunConfig(save_checkpoints_secs = 300,keep_checkpoint_max = 5)


# In[9]:


#Set model_fn
model_fn = cnnmodel_fn


# Set model directory.  This is to make sorting out runs more easily.

# In[10]:


name = 'cnn_model/cnn_model_'
name = name + 'dense' + str(model_params['dense_units']) + '_'
name = name + 'drop' + str(model_params['drop_out']) + '_'
name = name + 'lr' + str(model_params['learning_rate']) + '_'
name = name + time.strftime("%Y%m%d%H%M%S")
model_dir  = os.path.join('./',name)

print(model_dir)


# ### DO: Create tf.estimator

# In[11]:


estimator = tf.estimator.Estimator(
    model_fn=model_fn,model_dir=model_dir,params=model_params,config=config)


# Set parameters for input functions.

# In[12]:


train_files = get_ipython().getoutput('ls ./data/cifar10_data_00*.tfrecords')
val_files = get_ipython().getoutput('ls ./data/cifar10_data_01*.tfrecords')

train_params = {'filenames'    : train_files,
                'mode'         : tf.estimator.ModeKeys.TRAIN,
                'threads'      : 16,
                'shuffle_buff' : 100000,
                'batch'        : 100
               }

eval_params  = {'filenames'    : val_files,
                'mode'         : tf.estimator.ModeKeys.EVAL,
                'threads'      : 8,
                'batch'        : 200
               }


# ### DO: TrainSpec and EvalSpec
# 
# TranSpec:
# - Use train_params.
# - End after 20,000 steps.
# 
# EvalSpec:
# - Use eval_params.
# - 10 steps.
# - Eval executes every 60 seconds or training time (production will be much longer).

# In[13]:


train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset_input_fn(train_params),max_steps=20000)
eval_spec  = tf.estimator.EvalSpec(input_fn=lambda: dataset_input_fn(eval_params),steps=10,throttle_secs=60)


# ### DO: Run training and eval

# In[14]:


tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

