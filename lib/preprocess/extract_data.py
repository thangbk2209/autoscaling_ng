

from config import *
from lib.preprocess.extract_grid import GridExtractor


class DataExtractor:
    def __init__(self):
        pass

    def extract_grid(self):
        time_interval = [1800, 3600, 5400, 7200, 9000, 10800]
        file_data_path = CORE_DATA_DIR + '/{}/{}/{}.csv'.format('input_data', 'grid_data', 'anon_jobs')
        for _time_interval in time_interval:
            grid_extractor = GridExtractor(_time_interval, file_data_path)
            grid_extractor.compute_start_finish_point()
            grid_extractor.extract_information()
