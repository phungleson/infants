import tensorflow as tf

filenames = tf.train.string_input_producer(["2010.csv"])

reader = tf.TextLineReader()

key, value = reader.read(filenames)

# 246 values
record_defaults = [
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
  [1], [1], [1], [1], [1], [1],
]

tf.decode_csv(value, record_defaults=record_defaults)
