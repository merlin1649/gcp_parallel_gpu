import argparse
import tensorflow as tf
import time
import os
import numpy as np

def _conv(x,kernel,name):
    with tf.variable_scope(name):
        W = tf.get_variable(initializer=tf.truncated_normal(shape=kernel,stddev=0.01),name='W')
        b = tf.get_variable(initializer=tf.constant(0.0,shape=[kernel[3]]),name='b')
        conv = tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')
        activation = tf.nn.relu(tf.add(conv,b))
        pool = tf.nn.max_pool(activation,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return pool

def _dense(x,size_in,size_out,name,relu=False):
    with tf.variable_scope(name):
        flat = tf.reshape(x,[-1,size_in])
        W = tf.get_variable(initializer=tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
        b = tf.get_variable(initializer=tf.constant(0.0,shape=[size_out]),name='b')
        activation = tf.add(tf.matmul(flat,W),b)
        if relu==True:
            activation = tf.nn.relu(activation)
        return activation


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

# Data Pipline

def parse_tfrecord(example):
    feature={'idx'     : tf.FixedLenFeature((), tf.int64),
             'label'   : tf.FixedLenFeature((), tf.int64),
             'image'   : tf.FixedLenFeature((), tf.string, default_value="")}
    parsed = tf.parse_single_example(example, feature)
    image = tf.decode_raw(parsed['image'],tf.float64)
    image = tf.cast(image,tf.float32)
    image = tf.reshape(image,[32,32,3])
    return image, parsed['label']

def image_scaling(x):
    return tf.image.per_image_standardization(x)

def distort(x):
    x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
    x = tf.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x

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

def serving_input_receiver_fn():
  receiver_tensor = {'images': tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)}
  features = {'images': receiver_tensor['images']}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def run_app():
  config = tf.estimator.RunConfig(save_checkpoints_secs = 300,keep_checkpoint_max = 5)

  model_fn = cnnmodel_fn

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,model_dir=model_dir,params=model_params,config=config)

  train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset_input_fn(train_params),max_steps=20000)
  eval_spec  = tf.estimator.EvalSpec(input_fn=lambda: dataset_input_fn(eval_params),steps=10,throttle_secs=60)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  estimator.export_savedmodel(export_dir_base=model_dir,
                              serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
    '--model_dir',
    help='Model and checkpoint directory',
    required=True
    )
  args = parser.parse_args()
  arguments = args.__dict__
  model_dir = arguments.pop('model_dir')

  train_files = ['gs://gcpworkshop/data/cifar10_data_000.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_001.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_002.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_003.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_004.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_005.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_006.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_007.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_008.tfrecords',
                 'gs://gcpworkshop/data/cifar10_data_009.tfrecords']

  val_files   = ['gs://gcpworkshop/data/cifar10_data_010.tfrecords']

  model_params  = {'drop_out'      : 0.2,
                   'dense_units'   : 1024,
                   'learning_rate' : 1e-3
                  }

  train_params = {'filenames'    : train_files,
                  'mode'         : tf.estimator.ModeKeys.TRAIN,
                  'threads'      : 16,
                  'shuffle_buff' : 1000,
                  'batch'        : 100
                   }

  eval_params  = {'filenames'    : val_files,
                  'mode'         : tf.estimator.ModeKeys.EVAL,
                  'threads'      : 8,
                  'batch'        : 200
                 }

  run_app()


'''

Usage (non-Horovod):

python 04.5b-Horovod-Problem.py --model_dir='gs://gcpworkshop/models/tsaikevin000'

!! Make sure you have permissions to write to model_dir bucket !!

'''