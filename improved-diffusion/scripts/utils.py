import numpy as np
from config import *
from math import radians, cos, sin, asin, sqrt
import os
import pdb
import time
import json
import shutil
import logging
import hashlib
import math

train_ratio = 0.8
active_user_stay = 10
active_home_stay = 5
active_work_stay = 5
time_slot = 30  # 30 minutes

spatial_threshold = 0.005  # approximate to 500m
home_threshold = 0

work_start = 6
work_end = 23
work_spatial_threshold = 0.005
work_frequency_threshold = 0

max_stay_duration = 12
max_explore_range = 100  # multiply spatial_resolution: so that the max commute distance is 50km
spatial_resolution = 0.005 # 判定是两个基站间闪跳的空间距离阈值是500m
eta = 0.035
rho = 0.6
gamma = 0.21
performance_spatial_threshold = 0.005

max_grid = 180
GRID_SIZE = 1000
lon_l, lon_r, lat_b, lat_u = 115.43, 117.52, 39.44, 41.05 # Beijing
earth_radius = 6378137.0
pi = 3.1415926535897932384626
meter_per_degree = earth_radius * pi / 180.0
lat_step = GRID_SIZE * (1.0 / meter_per_degree)
ratio = np.cos((lat_b + lat_u) * np.pi / 360)
lon_step = lat_step / ratio

def map_ids_to_tokens_py(ids):
    loc = []
    for batch_id,batch in enumerate(ids):
        loc.append([])
        for grid_id in batch:
            x = grid_id.item() // max_grid
            y = grid_id.item() - x*max_grid - 1
            lon = (x+0.5)*lon_step + lon_l
            lat = (y+0.5)*lat_step + lat_b
            loc[batch_id].append([lon,lat])
    return loc

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance


def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def use_gpu(idx):
    # 0->2,3->1,1->3,2->0
    map = {0:2, 3:1, 1:3, 2:0}
    return map[idx]


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_gps(gps_file):
    with open(gps_file) as f:
        gpss = f.readlines()
    X = []
    Y = []
    for gps in gpss:
        x, y = float(gps.split()[0]), float(gps.split()[1])
        X.append(x)
        Y.append(y)
    return X, Y


def read_data_from_file(fp):
    """
    read a bunch of trajectory data from txt file
    :param fp:
    :return:
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def write_data_to_file(fp, dat):
    """Write a bunch of trajectory data to txt file.
    Parameters
    ----------
    fp : str
        file path of data
    dat : list
        list of trajs
    """
    with open(fp, 'w') as f:
        for i in range(len(dat)):
            line = [str(p) for p in dat[i]]
            line_s = ' '.join(line)
            f.write(line_s + '\n')


def read_logs_from_file(fp):
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[float(t) for t in tmp]]
    return np.asarray(dat, dtype='float')


def prep_workspace(workspace, datasets, oridata):
    """
    prepare a workspace directory
    :param workspace:
    :param oridata:
    :return:
    """
    data_path = '/data/stu/yangzeyu/trajgen'
    if not os.path.exists(data_path+'/%s/%s' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/data' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/data' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/logs' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/logs' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/figs' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/figs' % (datasets,workspace))
    '''
    shutil.copy("../data/%s/real.data" %
                oridata, "../%s/%s/data/real.data" % (datasets,workspace))
    shutil.copy("../data/%s/val.data" %
                oridata, "../%s/%s/data/val.data" % (datasets,workspace))
    shutil.copy("../data/%s/test.data" %
                oridata, "../%s/%s/data/test.data" % (datasets,workspace))
    shutil.copy("../data/%s/dispre_10.data" %
                oridata, "../%s/%s/data/dispre.data" % (datasets,workspace))
    '''
    with open(data_path+'/%s/%s/logs/loss.log' % (datasets,workspace), 'w') as f:
        pass

    with open(data_path+'/%s/%s/logs/jsd.log' % (datasets,workspace), 'w') as f:
        pass
    

def get_workspace_logger(datasets):
   
    data_path = '../data'  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s")
    fh = logging.FileHandler(data_path+'/%s/logs/all.log' % (datasets), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def merge(traj):
    #168长度的traj
    base_stamp = 1561910400

    new_traj = []
    traj_ = []
    for id_,i in enumerate(traj):
        start_hour = id_
        traj_.append([i,base_stamp+id_*3600])

    cnt = 0
    for id_,i in enumerate(traj_[1:]):
        if id_ == 0:
            new_traj.append(traj_[id_])
        if i[0] != traj_[id_][0]: # 如果格点id不一样
            new_traj[cnt].append(i[1])
            cnt += 1
            new_traj.append(i)
        if id_ == len(traj_[1:])-1:
            new_traj[cnt].append(base_stamp+7*24*3600)
     
    return new_traj

def stay_center(points):
    center = [0, 0]
    for p in points:
        center[0] += p[0]
        center[1] += p[1]
    center[0] /= len(points)
    center[1] /= len(points)
    return center

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def merge_same(user_trace):

    last_place = []
    user_pass = []
    for point in user_trace:
        if last_place and distance(last_place, point[0:2]) < spatial_threshold * 0.1:
            continue
        else:
            if last_place:
                user_pass[-1][3] = point[2]
            # stay: lat,lon,in_time,out_time
            stay = [point[0], point[1], point[2], point[2]]
            user_pass.append(stay)
            last_place = point[0:2]
    return user_pass

def merge_near(user_pass):
    merge_tmp = []
    user_stay = []
    for i, point in enumerate(user_pass):
        if len(merge_tmp) == 0:
            merge_tmp.append([point[0], point[1], i])
        else:
            flag = 0
            for p in merge_tmp:
                if distance(p[0:2], point[0:2]) < spatial_threshold:
                    merge_tmp.append([point[0], point[1], i])
                    flag = 1
                    break
            if flag:
                continue
            else:
                center = stay_center(merge_tmp)
                id1 = merge_tmp[0][2]
                id2 = merge_tmp[-1][2]
                ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
                user_stay.append(ele)
                merge_tmp = [[point[0], point[1], i]]
    if len(merge_tmp) > 0:
        center = stay_center(merge_tmp)
        id1 = merge_tmp[0][2]
        id2 = merge_tmp[-1][2]
        ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
        user_stay.append(ele)
    return user_stay

def stamp2array(time_stamp):
    return time.localtime(float(time_stamp))

def identify_home_(user_stay):
    #用时间戳的
    candidate = {}
    home_start = 19
    home_end = 9
    home_duration = float(24 - home_start + home_end)
    for p in user_stay:
        pid = str(p[0])
        duration = float(p[2] - p[1]) # sec
        start_time = stamp2array(str(p[1]))
        end_time = stamp2array(str(p[2]))
        start_hour = start_time.tm_hour
        end_hour = end_time.tm_hour
        r = 0
        #超出的算作1个stay，不满stay的按比例计入
        if start_hour > home_start: # 晚开始
            if end_hour < home_end: # 早结束，过0点
                r = min(1, duration / home_duration / 3600)
            else:
                if end_time.tm_mday != start_time.tm_mday: # 晚开始 晚结束 （跨天）
                    r = min(1, (24 - start_hour + home_end) / home_duration)
                else: # 未跨天
                    r = min(1, duration / home_duration / 3600)
        else:#早开始
            if end_hour <= home_end:#早结束，截断
                if end_time.tm_mday != start_time.tm_mday:
                    r = min(1, (24 - home_start + end_hour) / home_duration)
                else: # 同一天
                    r = min(1, duration / home_duration / 3600)
            else:
                if end_time.tm_mday == start_time.tm_mday: # 同一天
                    if start_hour < home_end and end_hour >= home_end: # 截前一段
                        r = min(1, (home_end - start_hour) / home_duration)
                    elif start_hour >= home_end and end_hour > home_start: # 截后一段
                        r = min(1, (end_hour - home_start) / home_duration)
                    elif start_hour < home_end and end_hour > home_start: #截两边
                        r = min(1, (end_hour - home_start + home_end - start_hour) / home_duration)
                else:# 早开始 晚结束
                    r = 1 # 包含一个完整的stay

        if pid in candidate:
            candidate[pid] += r
        else:
            candidate[pid] = r
    res = sorted(candidate.items(), key=lambda e: e[1], reverse=True)
    return res[0][0]
    

# def identify_home(user_stay):
#     # user_stay: 168长的id序列
#     home_start = 19
#     home_end = 9
#     home_duration = float(24 - home_start + home_end)
#     candidate = {}
#     for id_,p in enumerate(user_stay):
#         pid = p[0] # grid
#         start_time = p[1]
#         end_time = p[2]
#         duration = end_time - start_time
#         if duration<0: duration+=24
#         r = 0.
#         #超出的算作1个stay，不满stay的按比例计入

#         print(p)
#         if start_time >= home_start:#晚开始
#             if end_time <= home_end:#早结束
#                 r = min(1, duration / home_duration)
#                 print(1,r)

#             else:#晚结束，截断
#                 r = min(1, (24 - start_time + home_end) / home_duration)
#                 print(2,r)

#         else:#早开始
#             if end_time <= home_end:#早结束，截断
#                 r = min(1, (24 - home_start + end_time) / home_duration)
#                 print(3,r)

#             else:
#                 r = 1 # 包含一个完整的stay
#                 print(4)
#         if pid in candidate:
#             candidate[pid] += r
#         else:
#             candidate[pid] = r
#     res = sorted(candidate.items(), key=lambda e: e[1], reverse=True)
#     print(res)

#     # 返回home的grid id
#     return res[0][0]

def smooth_traces(user_stay, place):
    for i, p in enumerate(user_stay):
        if distance(p[0:2], place) < spatial_threshold:
            user_stay[i][0:2] = place
    return user_stay

def identify_work(user_stay, home):
    candidate = {}
    for p in user_stay:
        id = str(p[0]) + ',' + str(p[1])
        start_time = stamp2array(p[2])
        end_time = stamp2array(p[3])
        if start_time.tm_hour > work_start and end_time.tm_hour < work_end:
            if id in candidate:
                candidate[id] += 1
            else:
                candidate[id] = 1
    if len(candidate) == 0:
        return []
    for p in candidate:
        d = distance(home, [float(x) for x in p.split(',')])
        n = candidate[p]
        candidate[p] = [d, n]
    res = sorted(candidate.items(), key=lambda e: e[1][0] * e[1][1], reverse=True)
    return [float(x) for x in res[0][0].split(',')]

def date2stamp(time_date):
    time_array = time.strptime(time_date, "%Y%m%d%H%M%S")
    # time_array = time.strptime(time_date, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array))
    return time_stamp