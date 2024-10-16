import numpy as np
import json
import pickle
from config import *
import torch
import pdb
import copy
import os
from datetime import datetime
import pdb

args = create_argparser().parse_args()
torch.cuda.set_device(args.gpu)

class myDataset(object):
    def __init__(self, args, split:str):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grid = args.max_grid
        self.data_dir = args.data_dir
        self.split_data_dir = args.split_data_dir
        self.max_pos = args.max_pos
        # if not os.path.exists(os.path.join(self.split_data_dir,"train.pkl")):
        #     self.load_data()
        #     self.traj_preprocess()
        self.read_split_data(split)

    '''
    def load_data(self):
        def get_id_dict(data_dir):
            with open(data_dir,'r',encoding='utf-8') as f:
                dt_info = json.load(f)
            dt_id = {}
            for item in dt_info:
                dt_id[item["user_id"]] = item
            return dt_id
        print("loading data...")
        self.dt_info = get_id_dict(self.data_dir)

    def traj_preprocess(self):
        # 插值，以半小时为单位，一天内分为48个轨迹点
        def merge_day(traj,all_days,start_week):
            new_traj = [[[] for i in range(24)] for j in range(all_days)]#(天数,48)
            last = None
            for day in traj:
                for point in day:
                    if len(point) != 0:
                        last = point[0]
                        break
                break
            for i in range(all_days):
                for j in range(24):
                    if len(traj[i][j]) != 0:
                        last = traj[i][j][-1]    
                    new_traj[i][j] = copy.deepcopy(last)
                    new_traj[i][j].append(((i+start_week)%7))
            return new_traj

        self.traj = []
        for user_id, info in self.dt_info.items():#每个用户
            start = info["user_traces"][0][0][0:4]
            end = info["user_traces"][-1][0][0:4]
            start_week = datetime.strptime("2017" + start[0:4],"%Y%m%d").weekday()

            all_days = (int(end[1])-int(start[1]))*31 + int(end[2:4])-int(start[2:4]) + 1 #天数
            overall_traj = [[[] for i in range(24)] for j in range(all_days)]#(天数,24)

            for i in range(len(info["user_traces"])):#每个轨迹点
                time = info["user_traces"][i][0]
                week = datetime.strptime("2017" + time[0:4],"%Y%m%d").weekday()
                day_id = (int(time[0:4][1])-int(start[1]))*31 + int(time[0:4][2:4])-int(start[2:4])
                # hour_id = 2*int(time[4:6]) if int(time[6:])<30 else 2*int(time[4:6])+1
                hour_id = int(time[4:6])
                overall_traj[day_id][hour_id].append([info["user_traj"][i][0],
                                                        info["user_traj"][i][1]])
            self.traj.append(merge_day(overall_traj, all_days, start_week))

        self.traj_split = []
        for user in self.traj:
            temp = user[1:-1] #去头去尾
            days = len(temp)//14 #天数
            for i in range(days):
                self.traj_split.append(temp[(i*14):((i*14)+7)]) #取前7天
        
        #split
        self.n = len(self.traj_split)
        self.start_pos = {'train': 0, 'test': round(0.7 * self.n),
                        'val': round(0.9 * self.n)}
        
        temp = [i[1] for i in self.start_pos.items()]
        self.data_num = {
            'train': temp[1]-temp[0],
            'test': temp[2]-temp[1],
            'val': self.n-temp[2]
        }

        for i in self.data_num.keys():
            path = os.path.join(self.split_data_dir, i+'.pkl')
            pickle.dump(self.traj_split[self.start_pos[i]:self.start_pos[i]+self.data_num[i]], 
                    open(path,'wb'))
        print("data saved...")

        del self.dt_info
        del self.traj
        del self.traj_split
    '''

    def read_split_data(self,split:str):
        path = os.path.join(self.split_data_dir, split+'.pkl')
        self.split_data = pickle.load(open(path, 'rb'))
        print(split+" data loaded...")
    

    def __getitem__(self, id):
        item = np.zeros((self.max_pos+1,3), dtype=int)
        idx = 1 # 0为start token
        traj_data = self.split_data[id]['traj']

        for j in range(len(traj_data)): # 天数*24
            for i in range(24):
                if idx < self.max_pos + 1:
                    x = traj_data[j][i][0]
                    y = traj_data[j][i][1]
                    item[idx][0] = x * self.max_grid +y +1 # grid id
                    item[idx][1] = i +1 # hour
                    item[idx][2] = traj_data[j][i][2] +1 # y # week
                    idx += 1
                else:
                    break
        item = torch.tensor(item)
        item = item.to(self.device)

        return {"data":item,
                "length":len(traj_data)}

    def __len__(self):
        return len(self.split_data)


class Eval(object):
    # eval
    # 筛选、转化数据
    def __init__(self, args):
        super().__init__()
        print(args)

        self.data_dir = args
        # grid/ region_mapping
        self.GRID_SIZE = args.GRID_SIZE # Meter
        self.max_pos = args.max_pos
        self.data_dir = args.data_dir
        self.hw_dir = args.hw_dir
        self.preprocess()

    def preprocess(self):
        self.load_data()
        self.loc2grid()
        self.timestamp()
        self.get_data()
        self.write_data()

    def load_data(self):
        def get_id_dict(data_dir):
            with open(data_dir,'r',encoding='utf-8') as f:
                dt_info = json.load(f)
            dt_id = {}
            for item in dt_info:
                dt_id[item["user_id"]] = item
            return dt_id
        print("loading data...")
        self.dt_info = get_id_dict(self.data_dir)
        self.dt_hw = get_id_dict(self.hw_dir)
   
    def get_max_pos(self):
        self.max_pos = 0
        for user_id, term in self.dt_info.items():
            if len(term["user_traj"]) > self.max_pos:
                self.max_pos = len(term["user_traj"])
        print("self.max_pos=",self.max_pos)

    def loc2grid(self):
        def region_mapping(lon,lat,lon_l,lat_b,lon_step,lat_step):
            x = int((float(lon) - lon_l) // lon_step)
            y = int((float(lat) - lat_b) // lat_step)
            return [x, y]
        GRID_SIZE = self.GRID_SIZE

        lon_l, lon_r, lat_b, lat_u = 115.43, 117.52, 39.44, 41.05
        earth_radius = 6378137.0
        pi = 3.1415926535897932384626
        meter_per_degree = earth_radius * pi / 180.0
        lat_step = GRID_SIZE * (1.0 / meter_per_degree)
        ratio = np.cos((lat_b + lat_u) * np.pi / 360)
        lon_step = lat_step / ratio

        for user_id, term in self.dt_info.items():
            term["user_traj"] = []
            for i in term["user_traces"]:
                term["user_traj"].append(region_mapping(i[2],i[3],lon_l,lat_b,lon_step,lat_step))
            if self.dt_hw.get(user_id) != None:
                home = self.dt_hw[user_id]["user_home"]
                work = self.dt_hw[user_id]["user_work"]
                if len(home)!=0 and len(work)!=0 and work[0]!=None:
                    term["home"] = region_mapping(home[0], home[1],lon_l,lat_b,lon_step,lat_step)
                    term["work"] = region_mapping(work[0], work[1],lon_l,lat_b,lon_step,lat_step)

    def timestamp(self):
        def get_week(line):
            # 返回数字0—6，依次代表周一到周天
            time = "2017" + line[0:4]
            week = datetime.strptime(time,"%Y%m%d").weekday()
            return week

        #保留日、周几的信息
        #每天的时间转成48个格点
        for user_id, term in self.dt_info.items():
            for i in term["user_traces"]:
                # if int(i[0][6:])< 30:
                #     i[0] = i[0][0:6] + "00"
                # else:
                #     i[0] = i[0][0:6] + "30"
                i.append(get_week(i[0])) # 周几0-6

    def check_hw(self):
        check_home = []
        check_work = []
        for user_id, term in self.dt_info.items():
            home = 0
            work = 0
            for traj in term["user_traj"]:
                if traj == term["home"]:
                    home += 1
                elif traj == term["work"]:
                    work += 1
            if len(term["user_traj"]) != 0:
                check_home.append(float(home/len(term["user_traj"])))
                check_work.append(float(work/len(term["user_traj"])))
        check_home = np.mean(np.array(check_home), dtype=float)
        check_work = np.mean(np.array(check_work), dtype=float)
        print(check_home,check_work)
    
    def get_home_days(self):
        days_ = []
        for user_id, info in self.dt_info.items():
            home = info["home"]

            last_home = {}
            days = 0
            for id_, traj in enumerate(info["user_traj"]):
                if traj == home:
                    date = info["user_traces"][id_][0][0:4]
                    if last_home.get(date) == None:
                        last_home[date] = id_
                        days += 1
                    else:
                        last_home[date] = id_
            start = info["user_traces"][0][0][0:4]
            end = info["user_traces"][-1][0][0:4]
            all_days = (int(end[1])-int(start[1]))*31 + int(end[2:4])-int(start[2:4]) + 1 #天数
            days_.append([days/all_days])
        days_ = np.mean(np.array(days_))
        print(days_)

    def get_data(self):
        # [[all traj], [all time]]
        self.traj = []
        idx = 0
        for id_,info in self.dt_info.items():
            if info.get("home")!=None and info.get("work")!=None and len(info["user_traj"])!=0: # condition TODO
                self.traj.append([[],[]])
                self.traj[idx][0] = info["user_traj"] # grid traj
                for trace in info["user_traces"]:
                    self.traj[idx][1].append(trace[0]) # time
                idx += 1

    def write_data(self):
        def get_info(user_id):
            # 重组每个人的信息
            user_profile = self.dt_info[user_id]["user_profile"]
            user_traces = self.dt_info[user_id]["user_traces"]
            
            user_traj = self.dt_info[user_id]["user_traj"]
            home = None
            work = None
            if self.dt_info[user_id].get("home") != None and self.dt_info[user_id].get("work") != None:
                home = self.dt_info[user_id]["home"]
                work = self.dt_info[user_id]["work"]

            return {"user_id":user_id,
                "user_profile":user_profile,
                "user_traces":user_traces,
                "user_traj":user_traj,
                "home":home,
                "work":work}

        output_dir = "/data2/songyiwen/human_traj/dataset/processed_info.json"
        thres = 30.0 # cdf 20%
        uid2info_ = []
        count = 0
        for user_id, info in self.dt_info.items():
            if len(info["user_traces"])!=0:
                start = info["user_traces"][0][0][0:4]
                end = info["user_traces"][-1][0][0:4]
                all_days = (int(end[1])-int(start[1]))*31 + int(end[2:4])-int(start[2:4]) + 1
                points = float(len(info["user_traces"])/all_days)

            if points > thres:
                if info.get("home") != None and info.get("work") != None:
                    uid2info_.append(get_info(user_id))
                    count += 1
        with open(output_dir, 'w') as f:
            json.dump(uid2info_, f, indent=2, ensure_ascii=False)
        print("well done")
        print(count)


if __name__ == '__main__':
    us = myDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset=us, 
                                            batch_size=1,
                                            shuffle=False)
    # # us2 = Eval(args)

    for batch in dataloader:
        print(batch)
        exit()