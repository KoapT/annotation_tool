import numpy as np
import cv2
from scipy.optimize import least_squares

def apply_homography(H, pts):
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj

def ellipse_homo_error(params, object_points_2d, ellipse_params_obs, radii_real):
    # H: 8参数，ellipse_params: 5*N (cx,cy,rx,ry,theta)
    N = len(object_points_2d)
    H = np.append(params[:8], 1).reshape(3, 3)
    ellipse_params = params[8:].reshape(N, 5)
    errors = []
    # 1. 圆心单应误差
    proj_points = apply_homography(H, object_points_2d)
    for i in range(N):
        cx, cy, rx, ry, theta = ellipse_params[i]
        # 圆心误差
        errors.extend(proj_points[i] - [cx, cy])
        # 半径误差（长轴/短轴）：平面上圆心沿x/y轴移动真实半径，映射后与圆心距离应分别为rx, ry
        pt_plane = object_points_2d[i]
        r = radii_real[i]
        # 长轴方向
        pt_on_major = pt_plane + np.array([r, 0])
        proj_major = apply_homography(H, pt_on_major.reshape(1,2))[0]
        rx_proj = np.linalg.norm(proj_major - proj_points[i])
        errors.append(rx_proj - rx)
        # 短轴方向
        pt_on_minor = pt_plane + np.array([0, r])
        proj_minor = apply_homography(H, pt_on_minor.reshape(1,2))[0]
        ry_proj = np.linalg.norm(proj_minor - proj_points[i])
        errors.append(ry_proj - ry)
        # 椭圆角度误差（可选，通常忽略）
        # errors.append((theta - ellipse_params_obs[i, 4]) / 10.0)
    return np.array(errors)

def optimize(ellipse_label):
    # 1. 读取空间点和真实半径
    object_points = np.array([
        [-0.008, 0.0112, 0],
        [0.008, 0.0112, 0],
        [-0.016, 0, 0],
        [0, 0, 0],
        [0.016, 0, 0],
        [-0.008, -0.0139, 0],
        [0.008, -0.0139, 0],
    ], dtype=np.float32)
    object_points_2d = object_points[:, :2]
    radii_real = np.array([0.009, 0.009, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136]) / 2
    
    # 2. 读取像素椭圆
    ellipse_params_init = np.array(ellipse_label, dtype=np.float64)
    
    # 3. 单应矩阵初值（用圆心拟合）
    image_points_init = ellipse_params_init[:, :2]
    H_init, _ = cv2.findHomography(object_points_2d, image_points_init)
    
    # 4. 初始参数
    params0 = np.hstack([H_init.flatten()[:8], ellipse_params_init.flatten()])

    # 5. 优化
    result = least_squares(
        ellipse_homo_error, params0,
        args=(object_points_2d, ellipse_params_init, radii_real),
        verbose=2
    )
    N = len(object_points_2d)
    H_opt = np.append(result.x[:8], 1).reshape(3, 3)
    ellipse_params_opt = result.x[8:].reshape(N, 5)
    proj_points_opt = apply_homography(H_opt, object_points_2d)
    return proj_points_opt, ellipse_params_opt