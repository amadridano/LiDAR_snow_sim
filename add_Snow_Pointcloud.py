import os
import pandas
import logging
import numpy as np
import argparse
from tools.wet_ground.augmentation import ground_water_augmentation

from tools.snowfall.simulation import augment
from tools.snowfall.sampling import snowfall_rate_to_rainfall_rate, compute_occupancy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bins_dir",
    default="/home/amc-pc/snow_perception/synthetic_lidar/Emt_bins",
    type=str,
)

args = parser.parse_args()
mode = 'gunn'
snowfall_rate = 0.5
terminal_velocity = 0.2
noise_floor = 0.7
beam_divergence = 0.003
msg_number = 1

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    bins_dir = args.bins_dir 
    
    velo_filenames = sorted(os.listdir(bins_dir))
    
    bin_file_paths = [os.path.join(bins_dir, filename) for filename in velo_filenames]
    
    rain_rate = snowfall_rate_to_rainfall_rate(snowfall_rate, terminal_velocity)
    occupancy = compute_occupancy(snowfall_rate, terminal_velocity)
    
    snowflake_file_prefix = f'{mode}_{rain_rate}_{occupancy}'
    
    for file_path in bin_file_paths:

        pc = (np.fromfile(file_path, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.uint16)]))
        
        print(pc.shape[0])
        
        np_x = (np.array(pc['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc['intensity'], dtype=np.float32)).astype(np.float32)
        np_r = (np.array(pc['ring'], dtype=np.uint16)).astype(np.uint16)

        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i, np_r)))

        stats, points_32 = augment(pc=points_32, only_camera_fov=False,
                            particle_file_prefix=snowflake_file_prefix, noise_floor=noise_floor,
                            beam_divergence=float(np.degrees(beam_divergence)),
                            shuffle=True, show_progressbar=True)
        
        bin_file_name = str(msg_number).zfill(4)
        msg_number += 1
        
        output_bin_path = "/home/amc-pc/snow_perception/synthetic_lidar/" + bin_file_name + ".bin"  

        x = points_32[:, 0]
        y = points_32[:, 1]
        z = points_32[:, 2]
        intensity = points_32[:, 3]
        ring = points_32[:, 4]
        
        output_data = np.empty(len(points_32), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.uint16)
        ])

        output_data['x'] = x
        output_data['y'] = y
        output_data['z'] = z
        output_data['intensity'] = intensity
        output_data['ring'] = ring
        print(output_data.shape[0])
        output_data.tofile(output_bin_path)

        print(f"Los datos se han guardado correctamente en '{output_bin_path}'.")
        
