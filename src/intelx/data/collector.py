import csv
import time as pytime

class DataCollector:
    FIELDNAMES = [
        'env',
        'season',
        'episode',
        'step',
        'pixel_to_meter_ratio',
        'origin_x',
        'origin_y',
        'origin_theta',
        'query_x',
        'query_y',
        'query_theta',
        'origin_map',
        'query_map',
        'coverage',
        'goal_x',
        'goal_y',
        'origin_coverage',
        'timestamp'
    ]

    def __init__(self, log_path, env_name, origin, origin_map_fname, pixel_to_meter):
        """
        log_path: CSV file to write
        env_name: identifier for this map
        origin: (x, y, theta) of spawn in cells
        origin_map_fname: filename of the (sealed) map pgm
        pixel_to_meter: your RESOLUTION
        """
        self.env         = env_name
        self.origin      = origin
        self.origin_map  = origin_map_fname
        self.pix2m       = pixel_to_meter
        self.season      = 0
        self.episode     = 0
        self.step        = 0
        self.origin_cov  = 0.0  # before any exploration
        self.goal        = self.origin
        # open CSV
        self._f   = open(log_path, 'w', newline='')
        self._w   = csv.DictWriter(self._f, fieldnames=self.FIELDNAMES)
        self._w.writeheader()

    def start_episode(self, goal):
        """
        Call once after you sample your goal for this episode.
        goal: (x, y) of the goal in cells
        """
        self.episode    += 1
        self.step        = 0
        self.goal        = goal
        # reset origin coverage
        # self.origin_cov  = 0.0

    def log_step(self, query, query_map_fname, coverage):
        """
        Log one timestep.
        query: (x, y, theta) of robot this step in cell units
        query_map_fname: filename of saved local scan image
        coverage: float [0,1], explored_pixels / (window_px^2)
        """
        record = {
            'env':                   self.env,
            'season':                self.season,
            'episode':               self.episode,
            'step':                  self.step,
            'pixel_to_meter_ratio':  self.pix2m,
            'origin_x':              self.origin[0],
            'origin_y':              self.origin[1],
            'origin_theta':          self.origin[2],
            'query_x':               query[0],
            'query_y':               query[1],
            'query_theta':           query[2],
            'origin_map':            self.origin_map,
            'query_map':             query_map_fname,
            'coverage':              coverage,
            'goal_x':                self.goal[0],
            'goal_y':                self.goal[1],
            'origin_coverage':       self.origin_cov,
            'timestamp':             pytime.time()
        }
        self._w.writerow(record)
        self.step += 1

    def close(self):
        """Flush and close the CSV file."""
        self._f.close()
