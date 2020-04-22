import tensorflow as tf
from tensorflow.keras.layers import Lambda


def card_noble_mask(inputs):
  EPSILON = 0.00000001

  cards = inputs[0]
  mask = inputs[1]
  assert len(cards.shape) == 3, f'Got shape {cards.shape}'
  assert len(mask.shape) == 2, f'Got shape {mask.shape}'

  cards_count = tf.reduce_sum(mask, axis=1)
  cards_count = tf.expand_dims(cards_count, axis=-1)
  mask = tf.expand_dims(mask, axis=-1)
  cards = cards*mask
  cards_sum = tf.reduce_sum(cards, axis=1)
  cards_averages = tf.divide(cards_sum, (cards_count + EPSILON))

  assert len(cards_averages.shape) == 2, f'Got shape {cards_averages.shape}'

  return cards_averages

def tensor_squeeze(inputs):
  result = tf.squeeze(inputs, axis=1)
  return result

CardNobleMasking = Lambda(card_noble_mask)
TensorSqueeze = Lambda(tensor_squeeze)