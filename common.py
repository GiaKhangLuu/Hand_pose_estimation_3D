import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

def load_data_from_npz_file(file_path):
    data = np.load(file_path)
    return data

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_oak_2_rs_matrix(oak_r_raw, oak_t_raw, 
                        rs_r_raw, rs_t_raw):
    rs_r_raw = rs_r_raw.squeeze()
    rs_t_raw = rs_t_raw.squeeze()
    oak_r_raw = oak_r_raw.squeeze()
    oak_t_raw = oak_t_raw.squeeze()

    rs_r_mat = R.from_rotvec(rs_r_raw, degrees=False)
    rs_r_mat = rs_r_mat.as_matrix()

    oak_r_mat = R.from_rotvec(oak_r_raw, degrees=False)
    oak_r_mat = oak_r_mat.as_matrix()

    oak_r_t_mat = np.dstack([oak_r_mat, oak_t_raw[:, :, None]])
    rs_r_t_mat = np.dstack([rs_r_mat, rs_t_raw[:, :, None]])

    extra_row = np.array([0, 0, 0, 1] * oak_r_t_mat.shape[0]).reshape(-1, 4)[:, None, :]
    oak_r_t_mat = np.concatenate([oak_r_t_mat, extra_row], axis=1)
    rs_r_t_mat = np.concatenate([rs_r_t_mat, extra_row], axis=1)

    oak_r_t_mat_inv = np.linalg.inv(oak_r_t_mat)
    oak_2_rs_mat = np.matmul(rs_r_t_mat, oak_r_t_mat_inv)
    oak_2_rs_mat_avg = np.average(oak_2_rs_mat, axis=0)

    return oak_2_rs_mat_avg