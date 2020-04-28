from tensorboard.compat.tensorflow_stub.tensor_shape import TensorShape

x = ([TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 12]), TensorShape([None, 12]),
  TensorShape([None, 12]), TensorShape([None, 12]), TensorShape([None, 12]), TensorShape([None, 12]),
  TensorShape([None, 12]), TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]),
  TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 12]), TensorShape([None, 3]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 3]),
  TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]),
  TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 3]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]), TensorShape([None, 1]),
  TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]),
  TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 3]), TensorShape([None, 1]),
  TensorShape([None, 1]), TensorShape([None, 3])], TensorShape([None, 1]))

print(len(x))