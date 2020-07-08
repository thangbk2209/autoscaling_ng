
import pickle as pk

from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

from config import *


class DataReader:
    def __init__(self):
        self.google_trace_config = Config.GOOGLE_TRACE_DATA_CONFIG
        self.grid_data_config = Config.GRID_DATA_CONFIG
        self.traffic_data_config = Config.TRAFFIC_JAM_DATA_CONFIG
        self.normal_data_file = PROJECT_DIR + '/data/input_data/{}/normalized_data.pkl'.format(Config.DATA_EXPERIMENT)

    def __read_google_trace(self):
            time_interval = self.google_trace_config['time_interval']
            data_type = self.google_trace_config['data_type']
            data_name = self.google_trace_config['file_data_name'].format(data_type, time_interval)
            data_file_path = self.google_trace_config['data_path'].format(data_name)
            google_trace_df = read_csv(
                data_file_path, header=None, index_col=False, names=self.google_trace_config['colnames'],
                usecols=self.google_trace_config['usecols'], engine='python')
            cpu = google_trace_df['cpu_rate'].values.reshape(-1, 1)
            mem = google_trace_df['mem_usage'].values.reshape(-1, 1)
            disk_io_time = google_trace_df['disk_io_time'].values.reshape(-1, 1)
            disk_space = google_trace_df['disk_space'].values.reshape(-1, 1)

            official_data = {
                'cpu': cpu,
                'mem': mem,
                'disk_io_time': disk_io_time,
                'disk_space': disk_space
            }
            return official_data
    
    def __read_grid(self):
        time_interval = self.grid_data_config['time_interval']
        file_data_name = self.grid_data_config['file_data_name'].format(time_interval / 60)
        data_file_path = self.grid_data_config['data_path'].format(file_data_name)
        colnames = self.grid_data_config['colnames']
        df = read_csv(
            data_file_path, header=None, index_col=False, names=self.grid_data_config['colnames'], engine='python')
        jobs_id = df['job_id_data'].values.reshape(-1, 1)
        n_processes = df['n_proc_data'].values.reshape(-1, 1)
        used_cpu_time = df['used_cpu_time_data'].values.reshape(-1, 1)
        used_memory = df['used_memory_data'].values.reshape(-1, 1)
        users_id = df['user_id_data'].values.reshape(-1, 1)
        groups_id = df['group_id_data'].values.reshape(-1, 1)
        n_processes = n_processes[384:]

        official_data = {
            'jobs_id': jobs_id,
            'n_processes': n_processes,
            'used_cpu_time': used_cpu_time,
            'used_memory': used_memory,
            'users_id': users_id,
            'groups_id': groups_id
        }
        return official_data
    
    def __read_traffic(self):
        file_data_name = self.traffic_data_config['file_data_name']
        colnames = self.traffic_data_config['colnames']
        file_data_path = self.traffic_data_config['data_path'].format(file_data_name)

        df = read_csv(file_data_path, header=None, index_col=False, names=colnames, engine='python')
        traffic = df['megabyte'].values.reshape(-1, 1)

        official_data = {
            'eu': traffic
        }
        return official_data

    def read(self):
        if Config.DATA_EXPERIMENT == 'google_trace':
            data = self.__read_google_trace()
            return data
        elif Config.DATA_EXPERIMENT == 'grid':
            data = self.__read_grid()
            return data
        elif Config.DATA_EXPERIMENT == 'traffic':
            data = self.__read_traffic()
            return data
        else:
            print('>>> We do not support to experiment with this data <<<')
            return None, None
