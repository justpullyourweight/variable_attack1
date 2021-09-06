from utils import *

# slim = tf.contrib.slim


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = tf.pad(tensor=x, paddings=[[0, 0], [kern_size, kern_size], [kern_size, kern_size], [0, 0]], mode="CONSTANT")
    x = tf.nn.depthwise_conv2d(input=x, filter=stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    return x


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


def input_diversity(FLAGS, input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random.uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)  ## random_uniform()生成tensor
    rescaled = tf.image.resize(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 调整图像大小
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(tensor=rescaled, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    return tf.cond(pred=tf.random.uniform(shape=[1])[0] < tf.constant(FLAGS.prob), true_fn=lambda: padded,
                   false_fn=lambda: input_tensor)
    # tf.cond()类似于c语言中的if...else...


def norm_l1(a, b):
    return tf.sum(tf.abs(tf.subtract(a, b)))


def norm_l2(a, b):
    return tf.sqrt(tf.sum(tf.square(tf.subtract(a, b))))


def norm_li(a, b):
    # return torch.max(torch.abs(torch.sub(a, b)))
    return tf.abs(tf.subtract(a, b))
