Python Keras MNIST 手寫辨識
===========

Keras 實現 MNIST 手寫辨識
Python Keras MNIST 手寫辨識
 
介紹
====
深度學習領域中最經典的 Demo 就是 MNIST 手寫辨識，MNIST 資料即是由 28×28 灰階圖片，分別有 0~9 分佈 60,000 張訓練資料與 10,000 張測試資料 。
 
Python Keras MNIST 手寫辨識:這是一個神經網路的範例，利用了 Python Keras 來訓練一個手寫辨識分類 Model。
我們要的問題是將手寫數字的灰度影象（28×28 Pixel）分類為 10 類（0至9）。使用的資料集是 MNIST 經典資料集，它是由國家標準技術研究所（MNIST 的 NIST）
參考的程式碼
=======

https://github.com/samejack/blog-content/blob/master/keras-ml/mnist-neural-network.ipynb

images
======
from keras.datasets import mnist
 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

images 是用來訓練與測試的資料，label 則為每一筆影像資料對應的正確答案，每一張手寫圖片都是 28 x 28 的灰階 Bit Map，透過以下 Python 來看一下資料集的結構
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)
test_labels
輸出




建立準備訓練的神經網路
======
開始訓練神經網路以前，需要先建構網路，然後才開始訓練，如下：

from keras import models
from keras import layers
 
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

上面這裡是神經網路的核心組成方式，我們在全連線層建立了兩層，由一個有 512 個神經元的網路架構連線到 10 個神經元的輸出層。輸出層採用 softmax 表示數字 0~9 的機率分配，這 10 個數字的總和將會是 1。


建立的網路進行 compile
=======
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

上面的引數介紹的類神經網路基礎架構

以下將資料正規劃成為 0~1 的數值，變成 60000, 28×28 Shape 好送進上面定義的網路輸入層。

fix_train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
fix_test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255


開始訓練 MNIST 類神經網路
====

result = network.fit(
    fix_train_images,
    fix_train_labels,
    epochs=20,
    batch_size=128,
    validation_data=(fix_test_images, fix_test_labels)) 


執行結果如下：
====

test_loss, test_acc = network.evaluate(fix_test_images, fix_test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


透過 Keras 圖表協助分析訓練過程
====
以下方式可以繪製訓練過程 Loss Function 對應的損失分數。
Validation loss 不一定會跟隨 Training loss 一起降低，當 Model Over Fitting Train Data 時，就會發生 Validation loss 上升的情況。

history_dict = result.history
 
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
 
import matplotlib.pyplot as plt
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.show()

執行後的圖表如下：
====

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
 
plt.show()

參考資料
======
https://medium.com/bryanyang0528/deep-learning-keras-%E6%89%8B%E5%AF%AB%E8%BE%A8%E8%AD%98-mnist-b41757567684

https://ithelp.ithome.com.tw/articles/10186473
