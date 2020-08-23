import time
import math
import pickle as pkl
import threading

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
    def __init__(self, type_attr, min_val, max_val, range_val, name, fitness_type):

        self.type_attr = type_attr
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = range_val
        self.name = name
        self.fitness_type = fitness_type
        self.position = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(type_attr))
        self.position = self._corect_pos(self.position)

        self.velocity = np.random.uniform(-1, 1, len(type_attr))  # particle position

        self.pbest_position = self.position
        self.pbest_model = None
        self.pbest_value = float('inf')

    def _corect_pos(self, position):
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                position[i] = int(position[i])
        return position

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
        fitness, model = fitness_function(self.decode_position(self.position), self.fitness_type)
        # check to see if the current position is an individual best
        if fitness < self.pbest_value:
            self.pbest_position = self.position
            self.pbest_value = fitness
            self.pbest_model = model


class Space:
    def __init__(self, fitness_function, fitness_type, domain, num_particle=None):

        self.fitness_function = fitness_function
        self.fitness_type = fitness_type

        self._parse_domain(domain)
        self.num_particle = Config.NUM_PARTICLE
        self.create_particles()

        self.gbest_value = float('inf')
        self.gbest_model = None
        self.gbest_position = None
        self.gbest_attribute = None
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
                print(attr)
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
                Particle(self.type_attr, self.min_val, self.max_val, self.range_val, self.name, self.fitness_type))

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def evaluate(self, particle):
        particle.evaluate(self.fitness_function)

    def _set_gbest(self, particle):
        self.evaluate(particle)

        if self.gbest_value > particle.pbest_value:
            self.gbest_value = particle.pbest_value
            self.gbest_position = particle.pbest_position
            self.gbest_attribute = particle.decode_position(particle.pbest_position)
            self.gbest_model = particle.pbest_model
            self.gbest_paticle = particle

    def set_gbest(self):

        thread = []
        for particle in self.particles:
            _thread = threading.Thread(target=self._set_gbest, args=(particle,))
            thread.append(_thread)

        for _thread in thread:
            _thread.start()

        for _thread in thread:
            _thread.join()

        # for particle in self.particles:
            # self._set_gbest(particle)

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

    def save_best_particle(self, iteration, optimize_loss):
        model_name = self.gbest_model.model_path.split('/')[-1]

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            saved_path = f'{Config.RESULTS_SAVE_PATH}iter_{iteration + 1}'
        else:
            preprocess_name = self.gbest_model.preprocess_name
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/iter_{iteration + 1}'

        model_path = f'{saved_path}/model/{model_name}'
        gen_folder_in_path(saved_path)
        with open('{}/optimize_infor.pkl'.format(saved_path), 'wb') as out:
            pkl.dump(self.gbest_attribute, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(model_path, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(optimize_loss, out, pkl.HIGHEST_PROTOCOL)
        self.gbest_model.save_model(model_path)

    def optimize(self, max_iter, early_stopping=False, patience=20, step_save=2):
        optimize_loss = []

        for iteration in range(max_iter):
            self.w_old_velocation = (max_iter - iteration) / max_iter \
                * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation
            start_time = time.time()
            print(f'iteration: {iteration + 1}')

            self.set_gbest()

            self.move_particles()

            optimize_loss.append(self.gbest_value)
            training_history = 'fitness = %.8f with time for running: %.2f '\
                % (self.gbest_value, time.time() - start_time)
            print(training_history)

            # Save fitness, best model and best parameter
            if (iteration + 1) % step_save == 0:
                self.save_best_particle(iteration, optimize_loss)

            # if early_stopping:
            #     if len(optimize_loss) > patience:
            #         if early_stopping(optimize_loss, patience):
            #             print('[X] -> Early stoping because the profit is not increase !!!')
            #             break
        self.save_best_particle(iteration, optimize_loss)
        print('Best solution in iterations: {} has fitness = {}'.format(iteration + 1, self.gbest_value))
        return self.gbest_paticle
