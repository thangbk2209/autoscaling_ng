import time
import math
import pickle as pkl
import threading
import random

import numpy as np
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import multiprocessing
from multiprocessing import Pool
from queue import Queue

from lib.includes.utility import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.model_loader import ModelLoader


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

        self.pbest_value = float('inf')
        self.pbest_position = self.position
        self.pbest_attribute = self.decode_position(self.position)
        self.pbest_model = None

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
        print("My position: {} and pbest is: {}".format(self.pbest_position, self.pbest_value))

    def move(self):
        self.position = self.position + self.velocity
        self.position = self._corect_pos(self.position)
        self.position = np.clip(self.position, self.min_val, self.max_val)

    # evaluate current fitness
    def evaluate(self, fitness_function):
        fitness, model = fitness_function(self.decode_position(self.position), self.fitness_type)
        # check to see if the current position is an individual best
        if fitness < self.pbest_value:
            self.pbest_value = fitness
            self.pbest_position = self.position
            self.pbest_attribute = self.decode_position(self.position)
            self.pbest_model = model


class Space:
    def __init__(self, fitness_function, fitness_type, domain, num_particle=None):

        self.fitness_function = fitness_function
        self.fitness_type = fitness_type

        self._parse_domain(domain)
        self.num_particle = Config.NUM_PARTICLE
        self.data_preprocessor = DataPreprocessor()
        self.model_loader = ModelLoader()
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

    def _set_gbest(self, particle, optimize_option=True):

        if optimize_option:
            self.evaluate(particle)

        if self.gbest_value > particle.pbest_value:
            self.gbest_value = particle.pbest_value
            self.gbest_position = particle.pbest_position
            self.gbest_attribute = particle.decode_position(particle.pbest_position)
            self.gbest_model = particle.pbest_model
            self.gbest_paticle = particle


    def set_gbest(self, optimize_option=True):

        thread = []
        for particle in self.particles:
            _thread = threading.Thread(target=self._set_gbest, args=(particle, optimize_option,))
            thread.append(_thread)

        for _thread in thread:
            _thread.start()

        for _thread in thread:
            _thread.join()

        # for particle in self.particles:
            # self._set_gbest(particle, optimize_option)

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

    def save_best_particle(self):
        model_name = self.gbest_model.model_path.split('/')[-1]

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            saved_path = f'{Config.RESULTS_SAVE_PATH}best_model/iter_{self.iteration + 1}'
        else:
            preprocess_name = self.gbest_model.preprocess_name
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/best_model/iter_{self.iteration + 1}'

        model_path = f'{saved_path}/model/{model_name}'
        gen_folder_in_path(saved_path)
        with open('{}/optimize_infor.pkl'.format(saved_path), 'wb') as out:
            pkl.dump(self.gbest_value, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.gbest_position, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.gbest_attribute, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(model_path, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.iteration, out, pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.optimize_loss, out, pkl.HIGHEST_PROTOCOL)

        self.gbest_model.save_model(model_path)

    def save_current_state(self):

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            saved_path = f'{Config.RESULTS_SAVE_PATH}current_state'
        else:
            preprocess_name = self.gbest_model.preprocess_name
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/current_state'

        for i, particle in enumerate(self.particles):

            model_name = particle.pbest_model.model_path.split('/')[-1]
            model_path = f'{saved_path}/model_{i}/{model_name}'
            gen_folder_in_path(f'{saved_path}/model_{i}')
            with open('{}/model_{}/optimize_infor.pkl'.format(saved_path, i), 'wb') as out:
                # save particle best information
                pkl.dump(particle.pbest_value, out, pkl.HIGHEST_PROTOCOL)
                pkl.dump(particle.pbest_position, out, pkl.HIGHEST_PROTOCOL)
                pkl.dump(particle.pbest_attribute, out, pkl.HIGHEST_PROTOCOL)
                pkl.dump(model_path, out, pkl.HIGHEST_PROTOCOL)
                pkl.dump(self.iteration, out, pkl.HIGHEST_PROTOCOL)
                pkl.dump(self.optimize_loss, out, pkl.HIGHEST_PROTOCOL)

            # save particle best model
            particle.pbest_model.save_model(model_path)

    def load_current_particle_state(self):

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            saved_path = f'{Config.RESULTS_SAVE_PATH}current_state'
        else:
            preprocess_name = self.gbest_model.preprocess_name
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/current_state'

        if not os.path.exists(saved_path):
            self.optimize_loss = []
            self.iteration = -1
        else:
            self.particles = []
            dir_list = os.listdir(saved_path)
            for _dir in dir_list:
                dir_path = f'{saved_path}/{_dir}'

                with open(f'{dir_path}/optimize_infor.pkl', 'rb') as f:

                    particle_pbest_value = pkl.load(f)
                    particle_pbest_position = pkl.load(f)
                    particle_pbest_attribute = pkl.load(f)
                    model_path = pkl.load(f)
                    self.iteration = pkl.load(f)
                    self.optimize_loss = pkl.load(f)
                print('=== self.iteration in load ===')
                print(self.iteration)
                scaler_method = particle_pbest_attribute['scaler']
                scaler_method = Config.SCALERS[scaler_method - 1]
                sliding_encoder = particle_pbest_attribute['sliding_encoder']
                sliding_decoder = particle_pbest_attribute['sliding_decoder']

                batch_size = particle_pbest_attribute['batch_size']

                x_train_encoder, x_train_decoder, y_train_decoder, y_train, x_test_encoder, y_test, data_normalizer = \
                    self.data_preprocessor.init_data_bnn(sliding_encoder, sliding_decoder, scaler_method)

                validation_split = Config.VALID_SIZE
                n_train = int((1 - validation_split) * len(x_train_encoder))
                x_valid_encoder = x_train_encoder[n_train:]
                y_valid = y_train[n_train:]

                x_train_encoder = x_train_encoder[:n_train]
                y_train = y_train[:n_train]

                encoder_input_shape = [x_train_encoder.shape[1], x_train_encoder.shape[2]]
                decoder_input_shape = [x_train_decoder.shape[1], x_train_decoder.shape[2]]
                output_decoder_shape = [y_train_decoder.shape[1], y_train_decoder.shape[2]]
                output_shape = [y_train.shape[1]]
                model_path = model_path.split('data')

                model_path = CORE_DATA_DIR + model_path[-1]

                _particle = Particle(self.type_attr, self.min_val, self.max_val, self.range_val, self.name, self.fitness_type)
                _particle.pbest_value = particle_pbest_value
                _particle.pbest_position = particle_pbest_position
                _particle.pbest_attribute = particle_pbest_attribute
                _particle.pbest_model = self.model_loader.load_bnn(model_path, encoder_input_shape, output_shape)
                self.particles.append(_particle)

    def load_current_best_state(self):

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            saved_path = f'{Config.RESULTS_SAVE_PATH}best_model/iter_{self.iteration + 1}'
        else:
            preprocess_name = self.gbest_model.preprocess_name
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/best_model/iter_{self.iteration + 1}'

        if not os.path.exists(saved_path):
            self.optimize_loss = []
            self.iteration = -1
        else:

            with open('{}/optimize_infor.pkl'.format(saved_path), 'rb') as f:
                self.gbest_value = pkl.load(f)
                self.gbest_position = pkl.load(f)
                self.gbest_attribute = pkl.load(f)
                model_path = pkl.load(f)
                self.iteration = pkl.load(f)
                self.optimize_loss = pkl.load(f)

            scaler_method = self.gbest_attribute['scaler']
            scaler_method = Config.SCALERS[scaler_method - 1]
            sliding_encoder = self.gbest_attribute['sliding_encoder']
            sliding_decoder = self.gbest_attribute['sliding_decoder']

            batch_size = self.gbest_attribute['batch_size']

            x_train_encoder, x_train_decoder, y_train_decoder, y_train, x_test_encoder, y_test, data_normalizer = \
                self.data_preprocessor.init_data_bnn(sliding_encoder, sliding_decoder, scaler_method)

            validation_split = Config.VALID_SIZE
            n_train = int((1 - validation_split) * len(x_train_encoder))
            x_valid_encoder = x_train_encoder[n_train:]
            y_valid = y_train[n_train:]

            x_train_encoder = x_train_encoder[:n_train]
            y_train = y_train[:n_train]

            encoder_input_shape = [x_train_encoder.shape[1], x_train_encoder.shape[2]]
            decoder_input_shape = [x_train_decoder.shape[1], x_train_decoder.shape[2]]
            output_decoder_shape = [y_train_decoder.shape[1], y_train_decoder.shape[2]]
            output_shape = [y_train.shape[1]]
            model_path = model_path.split('data')

            model_path = CORE_DATA_DIR + model_path[-1]
            self.gbest_model = self.model_loader.load_bnn(model_path, encoder_input_shape, output_shape)

    def load_current_state(self):
        self.load_current_particle_state()
        # self.load_current_best_state()

    def optimize(self, max_iter, early_stopping=False, patience=20, step_save=1):

        self.load_current_state()
        self.set_gbest(optimize_option=False)
        print('==== best information after loading ===')
        print(self.gbest_attribute)
        print(self.gbest_position)
        print(self.gbest_value)
        current_iteration = self.iteration
        print('self.iteration: ', current_iteration)
        for iteration in range(current_iteration + 1, max_iter, 1):
            self.w_old_velocation = (max_iter - iteration) / max_iter \
                * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation

            start_time = time.time()
            print(f'iteration: {iteration + 1}')

            self.set_gbest()
            self.move_particles()
            self.optimize_loss.append(self.gbest_value)
            self.iteration = iteration
            print('==== best information ===')
            print(self.gbest_attribute)
            print(self.gbest_position)
            print(self.gbest_value)
            # Save fitness, best model and best parameter
            if (iteration + 1) % step_save == 0:
                self.save_current_state()
                self.save_best_particle()

            training_history = 'fitness = %.8f with time for running: %.2f '\
                % (self.gbest_value, time.time() - start_time)
            print(training_history)

            # if early_stopping:
            #     if len(optimize_loss) > patience:
            #         if early_stopping(optimize_loss, patience):
            #             print('[X] -> Early stoping because the profit is not increase !!!')
            #             break

        self.save_best_particle()
        print('Best solution in iterations: {} has fitness = {}'.format(self.iteration + 1, self.gbest_value))
        return self.gbest_paticle
