import numpy as np

principal_point_x, principal_point_y = 949.327192, 564.542811
camera_matrix = np.array([[1446.089410, 0, 949.327192],
                          [0, 1445.860418, 564.542811],
                          [0, 0, 1]])
rotation_matrix = np.array([[0.991905, -0.115870, 0.051955],
                            [-0.106303, -0.981477, -0.159384],
                            [0.069460, 0.152571, -0.985849]])
translation_vector = np.array([-0.242181, 0.200230, 1.818126])
dist_coeffs = np.array([-0.415581, 0.233585, -0.002746, 0.001573, 0.000000])
