import math

import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (len(norm_prev_pts) == 0):
        print('no prev points')
    elif (len(norm_prev_pts) == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ,prev_container, curr_container)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp,):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(curr_container.EM)
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ,prev_container, curr_container):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append([P[0], P[1], P[2]])
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    normal_pts = []
    for index in range(len(pts)):
        x, y = pts[index]
        x_normal = (x - pp[0]) / focal
        y_normal = (y - pp[1]) / focal
        normal_pts.append([x_normal, y_normal])
    return np.array(normal_pts)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnormal_pts = []
    for index in range(len(pts)):
        x_normal, y_normal = pts[index]
        x = (x_normal * focal) + pp[0]
        y = (y_normal * focal) + pp[1]
        unnormal_pts.append([x, y])
    return np.array(unnormal_pts)


def decompose(EM):
    EM = EM[:-1]
    # extract  R
    R = EM[:, :-1]
    translation = EM[:, -1]
    # extract foe
    foe = (translation[0] / translation[2], translation[1] / translation[2])
    # extract tZ
    tZ = translation[2]
    return R, foe, tZ


def rotate(pts, R):
    rotate_pts = []
    for point in pts:
        point_with_z = np.array([point[0], point[1], 1])
        a, b, c = np.dot(R, point_with_z)
        rotate_pts.append([a / c, b / c])
    return rotate_pts


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = ((foe[1] - p[1]) / (foe[0] - p[0]))
    n = (((p[1] * foe[0]) - (p[0] * foe[1])) / (foe[0] - p[0]))
    line = lambda x: m * x + n
    min_point_ind = 0
    min_destination = abs(
        (line(norm_pts_rot[min_point_ind][0]) - norm_pts_rot[min_point_ind][1]) / math.sqrt(m ** 2 + 1))
    for i in range(1, len(norm_pts_rot)):
        destination = abs((line(norm_pts_rot[i][0]) - norm_pts_rot[i][1]) / math.sqrt(m ** 2 + 1))
        if destination < min_destination:
            min_destination = destination
            min_point_ind = i
    return min_point_ind, norm_pts_rot[min_point_ind]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    by_x = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    by_y = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])
    x_dist = abs(p_curr[0] - p_rot[0])
    y_dist = abs(p_curr[1] - p_rot[1])
    z = (by_x * x_dist + by_y * y_dist) / (x_dist + y_dist)
    return z
