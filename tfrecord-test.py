import tensorflow as tf

for example in tf.python_io.tf_record_iterator("cards_train.tfrecord"):
    result = tf.train.Example.FromString(example)
    print(result)