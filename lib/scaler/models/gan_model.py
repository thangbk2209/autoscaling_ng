
import tensorflow as tf
import numpy as np

from lib.scaler.models.base_model import BaseModel
from lib.includes.utility import get_optimizer


class GanPredictor(BaseModel):
    def __init__(self, generator, discriminator, input_shape, output_shape, noise_shape, optimizer_g, optimizer_d,
                 num_train_d, loss_type, model_dir, is_wgan=False):
        self.generator = generator
        self.discriminator = discriminator
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.noise_shape = noise_shape
        self.loss_type = loss_type
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.num_train_d = num_train_d
        self.n_gen = 10
        self.c = 0.01
        # self._is_clip = False
        self.alpha = 1
        self.beta = 1
        self.gama = 1
        super().__init__(model_dir)

    def _build_model(self):
        self._x = tf.placeholder(tf.float32, [None] + self.input_shape, 'x')
        self._z = tf.placeholder(tf.float32, [self.n_gen, None] + self.noise_shape, 'noise')

        self._pred = self.generator(x=self._x, z=self._z[0])
        # self._pred = tf.reshape(self._pred, [-1] + self.output_shape)
        self._pred = tf.reshape(self._pred, [-1, 1, 1])

        self._y = tf.placeholder(tf.float32, self._pred.shape, 'y')

        x_fake = tf.concat([self._x, self._pred], axis=1, name='x_fake')
        x_real = tf.concat([self._x, self._y], axis=1, name='x_real')

        d_fake = self.discriminator(x_fake, reuse=False)
        d_real = self.discriminator(x_real, reuse=True)

        if self.loss_type == 'loss_gan':
            self._loss_g, self._loss_d = self._loss_gan(d_fake, d_real)
        elif self.loss_type == 'loss_gan_re':
            self._loss_g, self._loss_d = self._loss_gan_re(d_fake, d_real, self._pred, self._y)
        elif self.loss_type == 'loss_gan_re_d':
            self._loss_g, self._loss_d = self._loss_gan_re_d(d_fake, d_real, self._pred, self._y, self._x[:, -1, 0])
        else:
            raise NotImplementedError("Please check input loss function")

        d_vars, self._train_d = self._train_op(self._loss_d, self.optimizer_d, 3e-4, scope='discriminator')
        g_vars, self._train_g = self._train_op(self._loss_g, self.optimizer_g, 3e-4, scope='generator')

        # if self.is_wgan and self._is_clip:
        #     self.clip_d = [p.assign(tf.clip_by_value(p, -self.c, self.c)) for p in d_vars]

    def _loss_gan(self, d_fake, d_real):
        loss_d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        loss_d_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        loss_d = loss_d_real + loss_d_fake

        loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                    labels=tf.ones_like(
                                                        d_fake)))
        return loss_g, loss_d

    def _loss_wgan(self, d_fake, d_real):
        loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        loss_g = -tf.reduce_mean(d_fake)
        return loss_g, loss_d

    def _loss_gan_re(self, d_fake, d_real, predict, real):
        loss_g_gan, loss_d_gan = self._loss_gan(d_fake, d_real)
        # std_predict = tf.math.reduce_std(predicts, axis=0)

        loss_regression = tf.losses.mean_squared_error(real, predict)
        # loss_std = tf.reduce_mean(tf.square(std_predict))
        return self.alpha * loss_g_gan + self.beta * loss_regression, loss_d_gan

    def _loss_gan_re_d(self, d_fake, d_real, predict, real, xt):
        loss_g_gan, loss_d_gan = self._loss_gan(d_fake, d_real)
        loss_regression = tf.losses.mean_squared_error(real, predict)
        loss_sig = tf.abs(tf.sign(predict - xt) - tf.sign(real - xt))
        # loss_std = tf.reduce_mean(tf.square(std_predict))
        return self.alpha * loss_g_gan + self.beta * loss_regression + self.gama * loss_sig, loss_d_gan

    def _train_op(self, loss, optimizer, learning_rate, scope):
        optimizer = get_optimizer(optimizer, learning_rate)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        train_op = optimizer.minimize(loss, var_list=var_list)
        return var_list, train_op

    def _step(self, x, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._pred, feed_dict={self._x: x, self._z: self._get_noise(len(x))})
        elif mode == 'train':
            for i in range(self.num_train_d):
                ld, _ = self.sess.run([self._loss_d, self._train_d],
                                      {self._x: x, self._z: self._get_noise(len(x)), self._y: y})
                if self.is_wgan and self._is_clip:
                    self.sess.run(self.clip_d)
            lg, _ = self.sess.run([self._loss_g, self._train_g],
                                  {self._x: x, self._z: self._get_noise(len(x)), self._y: y})

    def _get_noise(self, batch_size, loc=0, scale=1):
        noise_shape = [self.n_gen, batch_size] + self.noise_shape
        return np.random.normal(loc=loc, scale=scale, size=noise_shape)
