import time
import math

import numpy as np
import random
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


class Particle:
    def __init__(self, graph=None, sess=None):

        self.graph = graph
        self.sess = sess

        self.x = self.graph.get_tensor_by_name('x:0')
        self.y = self.graph.get_tensor_by_name('y:0')
        self.prediction = self.graph.get_tensor_by_name('prediction:0')
        self.loss = self.graph.get_tensor_by_name('loss:0')

        self.trainable_variables = tf.trainable_variables()
        self.trainable_tensor = []
        # self.update_tensor_opts = []
        for i, v in enumerate(self.trainable_variables):
            v_tensor = self.graph.get_tensor_by_name(v.name)
            self.trainable_tensor.append(v_tensor)

        self.position = []
        self.velocity = []
        for v in tf.trainable_variables():
            v_value = self.sess.run(self.graph.get_tensor_by_name(v.name))
            self.position.append(v_value)
            self.velocity.append(np.zeros(v_value.shape))

        self.pbest_position = self.position
        self.velocity = np.array(self.velocity)
        self.pbest_position = np.array(self.pbest_position)
        self.position = np.array(self.position)
        self.pbest_value = float('inf')

    def __str__(self):
        print("My position: {} and pbest is: {}".format(self.graph, self.pbest_position))

    def fix_parameter_after_update(self):
        for i in range(len(self.position)):
            num_dimensional = len(self.position[i].shape)
            if num_dimensional == 1:
                for j in range(self.position[i].shape[0]):
                    if self.position[i][j] > 1:
                        self.position[i][j] = 1
                    elif self.position[i][j] < -1:
                        self.position[i][j] = -1
            elif num_dimensional == 2:
                for j in range(self.position[i].shape[0]):
                    for k in range(self.position[i].shape[1]):
                        if self.position[i][j][k] > 1:
                            self.position[i][j][k] = 1
                        elif self.position[i][j][k] < -1:
                            self.position[i][j][k] = -1

    def move(self):
        # self.fix_parameter_after_update()
        assignment_ops = []
        for i, v_tensor in enumerate(self.trainable_tensor):
            assignment_ops.append(tf.assign(v_tensor, self.position[i]))
        self.sess.run(assignment_ops)
        tf.reset_default_graph()

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.x: x})

    def fitness(self, x, y):
        fitness_value = self.sess.run(self.loss, feed_dict={self.x: x, self.y: y})
        fitness_value = np.around(fitness_value, decimals=7)
        return fitness_value


class Space:
    def __init__(self, num_particle=50, x_train=None, y_train=None, x_valid=None, y_valid=None, x_test=None,
                 y_test=None, batch_size=None, epochs=None):

        self.num_particle = num_particle
        self.particles = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_paticle = None
        self.max_w_old_velocation = 0.9
        self.min_w_old_velocation = 0.4
        self.w_local_best_position = 1.2
        self.w_global_best_position = 1.2

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def set_gbest(self):

        for particle in self.particles:
            fitness_cadidate = particle.fitness(self.x_train, self.y_train)
            if particle.pbest_value > fitness_cadidate:
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

            if self.gbest_value > fitness_cadidate:
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                self.gbest_paticle = particle

    def move_particles(self):
        for particle in self.particles:
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()

            change_base_on_old_velocity = self.w_old_velocation * particle.velocity
            change_base_on_local_best = self.w_local_best_position * r1 * (particle.pbest_position - particle.position)
            change_base_on_global_best = self.w_global_best_position * r2 * (self.gbest_position - particle.position)

            new_velocity = change_base_on_old_velocity + change_base_on_local_best + change_base_on_global_best

            particle.velocity = new_velocity
            # assign new value for trainables value to be nearer optimize global
            particle.position = particle.position + particle.velocity
            particle.move()

    def early_stopping(self, array, patience=20):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def check_most_n_value(self, cost_train_set):
        check = 0
        for i in range(len(cost_train_set) - 2, len(cost_train_set) - 6, -1):
            if cost_train_set[i] == cost_train_set[-1]:
                check += 1
            if check == 4:
                return True
        return False

    def train(self):
        cost_train_set = []
        cost_valid_set = []
        epoch_set = []
        epoch = 200
        for iteration in range(epoch):
            self.w_old_velocation = (epoch - iteration) / epoch * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation
            start_time = time.time()
            self.set_gbest()
            self.move_particles()
            cost_train_set.append(round(self.gbest_value, 5))
            cost_valid_set.append(self.gbest_paticle.fitness(self.x_valid, self.y_valid))
            training_history = 'iteration inference: %d fitness = %.8f with time for running: %.2f '\
                % (iteration, self.gbest_value, time.time() - start_time)
            print(training_history)
            if len(cost_train_set) > 20:
                if self.early_stopping(cost_train_set):
                    print('[X] -> Early stoping because the profit is not increase !!!')
                    break

        print("The best solution in iterations: {} has fitness = {} in train set".format(iteration, self.gbest_value))
        return self.gbest_paticle, self.gbest_paticle.predict(self.x_test), cost_train_set, cost_valid_set


class ParticleAutoEncoder:
    def __init__(self, graph=None, sess=None):

        self.graph = graph
        self.sess = sess

        self.x_encoder = self.graph.get_tensor_by_name('x_encoder:0')
        self.x_decoder = self.graph.get_tensor_by_name('x_decoder:0')
        self.y_decoder = self.graph.get_tensor_by_name('y_decoder:0')
        self.encoder_hidden_state = self.graph.get_tensor_by_name('encoder/hidden_state_encoder:0')
        self.prediction = self.graph.get_tensor_by_name('prediction:0')
        self.loss = self.graph.get_tensor_by_name('loss:0')
        self.trainable_variables = tf.trainable_variables()
        self.trainable_tensor = []
        for i, v in enumerate(self.trainable_variables):
            v_tensor = self.graph.get_tensor_by_name(v.name)
            self.trainable_tensor.append(v_tensor)

        self.position = []
        self.velocity = []
        for v in tf.trainable_variables():
            v_value = self.sess.run(self.graph.get_tensor_by_name(v.name))

            self.position.append(v_value)
            self.velocity.append(np.zeros(v_value.shape))
        self.pbest_position = self.position
        self.pbest_value = float('inf')

    def __str__(self):
        print("My position: {} and pbest is: {}".format(self.graph, self.pbest_position))

    def fix_parameter_after_update(self):
        # fix parameter bigger than 1 to 1 and smaller than -1 to -1
        for i in range(len(self.position)):
            num_dimensional = len(self.position[i].shape)
            if num_dimensional == 1:
                for j in range(self.position[i].shape[0]):
                    if self.position[i][j] > 1:
                        self.position[i][j] = 1
                    elif self.position[i][j] < -1:
                        self.position[i][j] = -1
            elif num_dimensional == 2:
                for j in range(self.position[i].shape[0]):
                    for k in range(self.position[i].shape[1]):
                        if self.position[i][j][k] > 1:
                            self.position[i][j][k] = 1
                        elif self.position[i][j][k] < -1:
                            self.position[i][j][k] = -1

    def print_position(self):
        for i in range(len(self.trainable_tensor)):
            print(self.sess.run(self.trainable_tensor[i]))

    def move(self):
        # self.fix_parameter_after_update()

        assignment_ops = []
        for i, v_tensor in enumerate(self.trainable_tensor):
            assignment_ops.append(tf.assign(v_tensor, self.position[i]))
        self.sess.run(assignment_ops)
        tf.reset_default_graph()

    def predict(self, x_encoder, x_decoder):
        return self.sess.run(self.prediction, feed_dict={self.x_encoder: x_encoder, self.x_decoder: x_decoder})

    def compute_state(self, x_encoder):
        return self.sess.run(self.encoder_hidden_state, feed_dict={self.x_encoder: x_encoder})

    def fitness(self, x_encoder, x_decoder, y_decoder):
        return self.sess.run(self.loss, feed_dict={self.x_encoder: x_encoder, self.x_decoder: x_decoder,
                                                   self.y_decoder: y_decoder})


class SpaceAutoEncoder:
    def __init__(self, num_particle=50, x_train_encoder=None, x_train_decoder=None, y_train_decoder=None,
                 x_valid_encoder=None, x_valid_decoder=None, y_valid_decoder=None, x_test_encoder=None,
                 x_test_decoder=None, y_test_decoder=None, batch_size=None, epochs=None):

        self.num_particle = num_particle
        self.particles = []
        self.x_train_encoder = x_train_encoder
        self.x_train_decoder = x_train_decoder
        self.y_train_decoder = y_train_decoder
        self.x_valid_encoder = x_valid_encoder
        self.x_valid_decoder = x_valid_decoder
        self.y_valid_decoder = y_valid_decoder
        self.x_test_encoder = x_test_encoder
        self.x_test_decoder = x_test_decoder
        self.y_test_decoder = y_test_decoder
        self.batch_size = batch_size
        self.epochs = epochs
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_paticle = None
        self.max_w_old_velocation = 0.9
        self.min_w_old_velocation = 0.4

        self.w_local_best_position = 1.2
        self.w_global_best_position = 1.2

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def set_gbest(self):
        # , batch_xs_encoder, batch_xs_decoder, batch_ys
        for particle in self.particles:
            fitness_cadidate = particle.fitness(self.x_train_encoder, self.x_train_decoder, self.y_train_decoder)

            if particle.pbest_value > fitness_cadidate:
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

            if self.gbest_value > fitness_cadidate:
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                self.gbest_paticle = particle

    def move_particles(self):
        for particle in self.particles:
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            new_velocity = np.add(np.add(np.multiply(self.w_old_velocation, particle.velocity),
                                         np.multiply(np.multiply(self.w_local_best_position, r1),
                                                     np.subtract(particle.pbest_position, particle.position))),
                                  np.multiply(np.multiply(r2, self.w_global_best_position),
                                              np.subtract(self.gbest_position, particle.position)))
            particle.velocity = new_velocity
            # assign new value for trainables value to be nearer optimize global
            particle.position = particle.position + particle.velocity
            particle.move()

    def early_stopping(self, array, patience=20):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def check_most_n_value(self, cost_train_set):
        check = 0
        for i in range(len(cost_train_set) - 2, len(cost_train_set) - 6, -1):
            if cost_train_set[i] == cost_train_set[-1]:
                check += 1
            if check == 4:
                return True
        return False

    def train(self):
        cost_train_set = []
        cost_valid_set = []
        epoch_set = []
        epoch = 200
        for iteration in range(epoch):
            self.w_old_velocation = (epoch - iteration) / epoch * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation
            # ratio = (epoch - iteration - 1) / (epoch - iteration)
            # self.w_global_best_position *= ratio
            # self.w_local_best_position *= ratio
            start_time = time.time()
            total_batch = int(len(self.x_train_encoder) / self.batch_size)

            self.set_gbest()
            self.move_particles()

            cost_train_set.append(round(self.gbest_value, 5))
            validation_loss = self.gbest_paticle.fitness(self.x_valid_encoder, self.x_valid_decoder,
                                                         self.y_valid_decoder)
            state_train = self.gbest_paticle.sess.run(self.gbest_paticle.encoder_hidden_state,
                                                      feed_dict={self.gbest_paticle.x_encoder: self.x_train_encoder})
            cost_valid_set.append(validation_loss)
            training_history = 'iteration autoencoder: %d, train loss = %.10f, validation loss = %.10f with time for running: %.2f'\
                % (iteration + 1, self.gbest_value, validation_loss, time.time() - start_time)
            print(training_history)
            if len(cost_train_set) > 20:
                if self.early_stopping(cost_train_set):
                    print('[X] -> Early stoping because the profit is not increase !!!')
                    break

        print("The best solution in iterations: {} has fitness = {} in train set".format(iteration, self.gbest_value))
        return self.gbest_paticle, self.gbest_value, cost_train_set, cost_valid_set
