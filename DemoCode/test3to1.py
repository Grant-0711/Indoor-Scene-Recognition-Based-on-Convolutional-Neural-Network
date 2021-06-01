from PIL import Image,ImageChops,ImageEnhance
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage import io, transform
w = 100
h = 100
c = 3
if __name__ == "__main__":
    img_data = io.imread('airport_inside_0001.jpg')
    # img_data = Image.open('airport_inside_0001.jpg', 'r')
    img_data = transform.resize(img_data, (w, h,c))
    print(img_data.size)	# (1000,625)
    
    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(img_data)

    img = np.array(img_data)
    img = img.reshape(30000,30000,1)
    img_tensor = tf.convert_to_tensor(img)
    img_tensor = tf.image.grayscale_to_rgb(img_tensor)

    sess = tf.Session()
    img = sess.run(img_tensor)
    print(img_tensor.shape)	# (625,1000,3)

    plt.subplot(1, 2, 2)
    plt.title("rgb")
    plt.imshow(img)
    plt.show()
