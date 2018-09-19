import tensorflow as tf
import Option

LR    = Option.lr
bitsW = Option.bitsW
bitsA = Option.bitsA
bitsG = Option.bitsG
bitsE = Option.bitsE
bitsR = Option.bitsR
L2 = Option.L2

Graph = tf.get_default_graph()

def S(bits):
  return 2.0 ** (bits - 1)

def Shift(x):
  return 2 ** tf.round(tf.log(x) / tf.log(2.0))

def C(x, bits=32):
  if bits > 15 or bits == 1:
    delta = 0.
  else:
    delta = 1. / S(bits)
  MAX = +1 - delta
  MIN = -1 + delta
  x = tf.clip_by_value(x, MIN, MAX, name='saturate')
  return x

def Q(x, bits):
  if bits > 15:
    return x
  elif bits == 1:  # BNN
    return tf.sign(x)
  else:
    SCALE = S(bits)
    return tf.round(x * SCALE) / SCALE

def W(x,scale = 1.0):
  with tf.name_scope('QW'):
    y = Q(C(x, bitsW), bitsW)
    # we scale W in QW rather than QA for simplicity
    if scale > 1.8:
      y = y/scale
    # if bitsG > 15:
      # when not quantize gradient, we should rescale the scale factor in backprop
      # otherwise the learning rate will have decay factor scale
      # x = x * scale
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize and Clip

def A(x):
  with tf.name_scope('QA'):
    x = C(x, bitsA)
    y = Q(x, bitsA)
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize, but keep Clip

def G(x):
  with tf.name_scope('QG'):
    if bitsG > 15:
      return x
    else:
      if x.name.lower().find('batchnorm') > -1:
        return x  # batch norm parameters, not quantize now

      tf.summary.histogram("g-"+x.name, x)
      xmax = tf.reduce_max(tf.abs(x))
      tf.summary.scalar("g-max-"+x.name, xmax)
      x = x / Shift(xmax)
      tf.summary.histogram("g-shifted-"+x.name, x)

      norm = Q(LR * x, bitsR)
      tf.summary.histogram("g-norm-"+x.name, norm)

      norm_sign = tf.sign(norm)
      norm_abs = tf.abs(norm)
      norm_int = tf.floor(norm_abs)
      norm_float = norm_abs - norm_int
      rand_float = tf.random_uniform(x.get_shape(), 0, 1)
      norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )

      ret = norm / S(bitsG)
      tf.summary.histogram("g-quant-"+x.name, ret)
      return ret

@tf.RegisterGradient('Error')
def error(op, x):
  if bitsE > 15:
    return x
  else:
    tf.summary.histogram("back-"+(x.name), x)
    xmax = tf.reduce_max(tf.abs(x))
    tf.summary.scalar("back-max-"+(x.name), xmax)
    xmax_shift = Shift(xmax)
    tf.summary.scalar("back-xmaxshift-"+(x.name), xmax_shift)
    ret = Q(C( x /xmax_shift, bitsE), bitsE)
    tf.summary.histogram("quant-back-"+(x.name), ret)
    return ret

def E(x):
  with tf.name_scope('QE'):
    with Graph.gradient_override_map({'Identity': 'Error'}):
      return tf.identity(x)

