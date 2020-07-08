import time
import math

import numpy as np
import random
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import multiprocessing
from multiprocessing import Pool
from queue import Queue

from lib.includes.utility import *


class Particle:
    def __init__(self, type_attr, min_val, max_val, range_val, name):

        self.type_attr = type_attr
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = range_val
        self.name = name
        self.position = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(type_attr))
        self.position = self._corect_pos(self.position)

        self.velocity = np.random.uniform(-1, 1, len(type_attr))  # particle position

        self.pbest_position = self.position
        self.pbest_value = float('inf')

    def _corect_pos(self, position):
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                position[i] = int(position[i])
        return position

    # def decode_position(self, position):
    #     result = []
    #     for i, t in enumerate(self.type_attr):
    #         if t == 'discrete':
    #             result.append(self.range_val[i][int(position[i])])
    #         else:
    #             result.append(position[i])

    #     return result

    def decode_position(self, position):
        result = {}
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                result[self.name[i]] = self.range_val[i][int(position[i])]
            else:
                result[self.name[i]] = position[i]
        return result

    def __str__(self):
        print("My position: {} and pbest is: {}".format(self.graph, self.pbest_position))

    def move(self):
        self.position = self.position + self.velocity
        self.position = self._corect_pos(self.position)
        self.position = np.clip(self.position, self.min_val, self.max_val)

    # evaluate current fitness
    def evaluate(self, fitness_function):
        self.err = fitness_function(self.decode_position(self.position))

        # check to see if the current position is an individual best
        if self.err < self.pbest_value:
            self.pbest_position = self.position
            self.pbest_value = self.err


class Space:
    def __init__(self, fitness_function, domain, num_particle=5):

        self.fitness_function = fitness_function

        self._parse_domain(domain)
        self.num_particle = num_particle
        self.create_particles()

        self.gbest_value = float('inf')
        self.gbest_paticle = None
        self.max_w_old_velocation = 0.9
        self.min_w_old_velocation = 0.4
        self.w_local_best_position = 1.2
        self.w_global_best_position = 1.2

    def _parse_domain(self, domain):
        name = []
        type_attr = []
        max_val = []
        min_val = []
        range_val = []

        for attr in domain:
            name.append(attr['name'])
            type_attr.append(attr['type'])
            if attr['type'] == 'discrete':
                min_val.append(0)
                max_val.append(len(attr['domain']) - 1)
            elif attr['type'] == 'continuous':
                min_val.append(attr['domain'][0])
                max_val.append(attr['domain'][1])
            range_val.append(attr['domain'])

        self.name = name
        self.type_attr = type_attr
        self.max_val = np.array(max_val)
        self.min_val = np.array(min_val)
        self.range_val = range_val

    def decode_position(self, position):
        result = {}
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                result[self.name[i]] = self.range_val[int(position[i])]
            else:
                result[self.name[i]] = position[i]
        return result

    def create_particles(self):
        self.particles = []
        for i in range(self.num_particle):
            self.particles.append(
                Particle(self.type_attr, self.min_val, self.max_val, self.range_val, self.name))

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
        
    def evaluate(self, particle):
        return particle.evaluate(self.fitness_function)

    def set_gbest(self):

        # queue = Queue()
        # for particle in self.particles:
        #     queue.put_nowait(particle)
            # summary = open(self.evaluation_path, 'a+')
            # summary.write('Model, MAE, RMSE\n')
            # print('>>> start experiment ANN model <<<')
        # pool = Pool(1)
        # pool.map(self.evaluate, list(queue.queue))
        # pool.close()
        # pool.join()
        # pool.terminate()

        for particle in self.particles:
            particle.evaluate(self.fitness_function)
            # print(particle.pbest_value)
            if particle.pbest_value < self.gbest_value:
                self.gbest_position = particle.pbest_position
                self.gbest_value = particle.pbest_value

            if self.gbest_value > particle.pbest_value:
                self.gbest_value = particle.pbest_value
                self.gbest_position = particle.pbest_position
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
            particle.move()

    def optimize(self, max_iter, early_stopping, patience=20):
        optimize_loss = []

        for iteration in range(max_iter):
            self.w_old_velocation = (max_iter - iteration) / max_iter \
                * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation
            start_time = time.time()

            self.set_gbest()

            self.move_particles()

            optimize_loss.append(round(self.gbest_value, 7))
            training_history = 'iteration: %d fitness = %.8f with time for running: %.2f '\
                % (iteration, self.gbest_value, time.time() - start_time)
            print(training_history)
            if early_stopping:
                if len(optimize_loss) > patience:
                    if early_stopping(optimize_loss, patience):
                        print('[X] -> Early stoping because the profit is not increase !!!')
                        break

        print('The best solution in iterations: {} has fitness = {} in train set'.format(iteration, self.gbest_value))
        return self.gbest_paticle
