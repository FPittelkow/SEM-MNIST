# C Friedrich Pittelkow
# f.pittelkow@gmail.com
# 2016

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

n_example = 10000

i_show = 0
v = False
vw = False

batch_size = 100
steps = 1000

l = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = l.datasets.load_dataset('mnist')
images = mnist.train.images

labels = np.asarray(mnist.train.labels, dtype=np.int32)

test_data = mnist.test.images
test_labels = np.asanyarray(mnist.test.labels, dtype=np.int32)

images = images[:n_example]
labels = labels[:n_example]


def visualize_sample(i):
    img = test_data[i]
    plt.title('Bsp. %d. Typ: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()

if v:
    visualize_sample(i_show)

vec_feature= l.infer_real_valued_columns_from_input(images)

classifier = l.LinearClassifier(n_classes=10, feature_columns=vec_feature)
classifier.fit(images, labels, batch_size=batch_size, steps=steps)

def predict(i):
    print("Vermutet: %d, Typ: %d" % (classifier.predict(test_data[i]), test_labels[i]))

predict(8)

def evacc():
    classifier.evaluate(test_data, test_labels)
    print(classifier.evaluate(test_data, test_labels)["accuracy"])

evacc()

def showWeights():
    weights = classifier.weights_
    f, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.reshape(-1)

    for i in range(len(axes)):
        a = axes[i]
        a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)

        a.set_title(i)
        a.set_xticks(())
        a.set_yticks(())

    plt.show()

if vw:
    showWeights()