import tensorflow as tf
import numpy as np

#声明特征列，这里声明了一个实值特征。
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#一个estimator是调用训练（拟合）和评估（推理）的前端。提供很多预#定义的类型，如线性回归，逻辑回归，线性分类，逻辑分类和许多神经#网络分类器和回归。下面是linear regression：
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

#TensorFlow提供了许多帮助方法来读取和设置数据集。
#这里我们使用两个数据集：一个用于训练，一个用于评估
#我们必须告诉功能我们想要多少batch的数据（num_epochs），每个#batch应该有多大。

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
batch_size=4,                                             num_epochs=1000)

eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

#我们可以通过调用方法和传递训练数据集来迭代1000次训练步骤。
estimator.fit(input_fn=input_fn, steps=1000)

#验证模型误差
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
