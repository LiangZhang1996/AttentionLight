import os
import pandas as pd
import numpy as np
import json
import shutil
import copy
from collections import deque
import matplotlib.pyplot as plt


def get_metrics(duration_list, traffic_name, total_summary_metrics, num_of_out):
    # calculate the mean final 10 rounds
    validation_duration_length = 10
    duration_list = np.array(duration_list)
    validation_duration = duration_list[-validation_duration_length:]
    validation_through = num_of_out[-validation_duration_length:]
    final_through = np.round(np.mean(validation_through), decimals=2)
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    total_summary_metrics["traffic"].append(traffic_name.split(".json")[0])
    total_summary_metrics["final_duration"].append(final_duration)
    total_summary_metrics["final_duration_std"].append(final_duration_std)
    total_summary_metrics["final_through"].append(final_through)

    return total_summary_metrics


def summary_detail_RL(memo_rl, total_summary_rl):
    """
    Used for test RL results
    """
    records_dir = os.path.join("records", memo_rl)
    for traffic_file in os.listdir(records_dir):
        if ".json" not in traffic_file:
            continue
        # print(traffic_file)

        traffic_env_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')
        dic_traffic_env_conf = json.load(traffic_env_conf)
        run_counts = dic_traffic_env_conf["RUN_COUNTS"]
        num_intersection = dic_traffic_env_conf['NUM_INTERSECTIONS']
        duration_each_round_list = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []
        test_round_dir = os.path.join(records_dir, traffic_file, "test_round")
        try:
            round_files = os.listdir(test_round_dir)
        except:
            print("no test round in {}".format(traffic_file))
            continue
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        for round_rl in round_files:
            df_vehicle_all = []
            for inter_index in range(num_intersection):
                try:
                    round_dir = os.path.join(test_round_dir, round_rl)
                    df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                   sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                   names=["vehicle_id", "enter_time", "leave_time"])
                    # [leave_time_origin, leave_time, enter_time, duration]
                    df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                    df_vehicle_inter['leave_time'].fillna(run_counts, inplace=True)
                    df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - \
                                                   df_vehicle_inter["enter_time"].values
                    tmp_idx = []
                    for i, v in enumerate(df_vehicle_inter["vehicle_id"]):
                        if "shadow" in v:
                            tmp_idx.append(i)
                    df_vehicle_inter.drop(df_vehicle_inter.index[tmp_idx], inplace=True)

                    ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                    print("------------- inter_index: {0}\tave_duration: {1}".format(inter_index, ave_duration))
                    df_vehicle_all.append(df_vehicle_inter)
                except:
                    print("======= Error occured during reading vehicle_inter_{}.csv")

            if len(df_vehicle_all) == 0:
                print("====================================EMPTY")
                continue

            df_vehicle_all = pd.concat(df_vehicle_all)
            # calculate the duration through the entire network
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()  # mean amomng all the vehicle

            duration_each_round_list.append(ave_duration)

            num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
            num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

            print("==== round: {0}\tave_duration: {1}\tnum_of_vehicle_in:{2}\tnum_of_vehicle_out:{2}"
                  .format(round_rl, ave_duration, num_of_vehicle_in[-1], num_of_vehicle_out[-1]))
            duration_flow = vehicle_duration.reset_index()
            duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x: x.split('_')[1])
            duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
            print(duration_flow_ave)
        result_dir = os.path.join("summary", memo_rl, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        total_summary_rl = get_metrics(duration_each_round_list, traffic_file, total_summary_rl, num_of_vehicle_out)
        total_result = pd.DataFrame(total_summary_rl)
        total_result.to_csv(os.path.join("summary", memo_rl, "total_test_results.csv"))


def summary_detail_conventional(memo_cv):
    """
    Used for test conventional results.
    """
    total_summary_cv = []
    records_dir = os.path.join("records", memo_cv)
    for traffic_file in os.listdir(records_dir):
        if "anon" not in traffic_file:
            continue
        traffic_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')

        dic_traffic_env_conf = json.load(traffic_conf)
        run_counts = dic_traffic_env_conf["RUN_COUNTS"]

        print(traffic_file)
        train_dir = os.path.join(records_dir, traffic_file)
        use_all = True
        if use_all:
            with open(os.path.join(records_dir, traffic_file, 'agent.conf'), 'r') as agent_conf:
                dic_agent_conf = json.load(agent_conf)

            df_vehicle_all = []
            NUM_OF_INTERSECTIONS = int(traffic_file.split('_')[1]) * int(traffic_file.split('_')[2])

            for inter_id in range(int(NUM_OF_INTERSECTIONS)):
                vehicle_csv = "vehicle_inter_{0}.csv".format(inter_id)

                df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, vehicle_csv),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                # [leave_time_origin, leave_time, enter_time, duration]
                df_vehicle_inter_0['leave_time_origin'] = df_vehicle_inter_0['leave_time']
                df_vehicle_inter_0['leave_time'].fillna(run_counts, inplace=True)
                df_vehicle_inter_0['duration'] = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0[
                    "enter_time"].values

                tmp_idx = []
                for i, v in enumerate(df_vehicle_inter_0["vehicle_id"]):
                    if "shadow" in v:
                        tmp_idx.append(i)
                df_vehicle_inter_0.drop(df_vehicle_inter_0.index[tmp_idx], inplace=True)

                ave_duration = df_vehicle_inter_0['duration'].mean(skipna=True)
                print("------------- inter_index: {0}\tave_duration: {1}".format(inter_id, ave_duration))
                df_vehicle_all.append(df_vehicle_inter_0)

            df_vehicle_all = pd.concat(df_vehicle_all, axis=0)
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()
            num_of_vehicle_in = len(df_vehicle_all['vehicle_id'].unique())
            num_of_vehicle_out = len(df_vehicle_all.dropna()['vehicle_id'].unique())
            save_path = os.path.join('records', memo_cv, traffic_file).replace("records", "summary")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # duration.to_csv(os.path.join(save_path, 'flow.csv'))
            total_summary_cv.append(
                [traffic_file, ave_duration, num_of_vehicle_in, num_of_vehicle_out, dic_agent_conf["FIXED_TIME"]])
        else:
            shutil.rmtree(train_dir)
    total_summary_cv = pd.DataFrame(total_summary_cv)
    total_summary_cv.sort_values([0], ascending=[True], inplace=True)
    total_summary_cv.columns = ['TRAFFIC', 'DURATION', 'CAR_NUMBER_in', 'CAR_NUMBER_out', 'CONFIG']
    total_summary_cv.to_csv(os.path.join("records", memo_cv,
                                         "total_baseline_results.csv").replace("records", "summary"),
                            sep='\t', index=False)


def get_att_eng(memo_rl):
    """
    ATT from eng
    """
    records_dir = os.path.join("records", memo_rl)
    traffic_files = os.listdir(records_dir)
    traffic_files.sort()
    att_list = []
    for traffic_file in traffic_files:
        if ".json" not in traffic_file:
            continue        
        test_round_dir = os.path.join(records_dir, traffic_file, "test_round")
        try:
            round_files = os.listdir(test_round_dir)
        except:
            print("no test round in {}".format(traffic_file))
            continue
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        att_all=[]
        for round_rl in round_files:
            round_dir = os.path.join(test_round_dir, round_rl)
            df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_0.csv"),
                                            sep=',', header=0, dtype={0: str, 1: float, 2: float, 3: float},
                                            names=["vehicle_id", "enter_time", "leave_time", "eng_att"])
            att = df_vehicle_inter['eng_att'][0]
            att_all.append(att)
        last_10_att = att_all[-10:]
        mean_att = np.mean(last_10_att)
        att_list.append(mean_att)
        print(f'mean_att: {mean_att:.2f} {traffic_file}')


def extract_test_att(filepath):
    """
    解析单个实验中一张地图的结果，提取 att 值，保存到 csv 文件
    filepath: 实验结果路径. 如：records/att_hz_8_6/anon_4_4_hangzhou_real_5816.json_11_27_09_33_48
    """
    max_episode = 80
    att_de = deque(maxlen=max_episode)
    if filepath.endswith("test_round"):
        testpath = filepath
    else:
        testpath = os.path.join(filepath, "test_round")
    for i in range(max_episode):
        att_filepath = os.path.join(testpath, "round_{}".format(i), "eng-att.txt")
        with open(att_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                att =  round(float(line),2)
                att_de.append(att)
    print(f"extract att from {testpath}, len={len(att_de)}")
    # save att_de to csv
    csv_path = os.path.join(filepath, "att.csv")
    with open(csv_path, "w") as f:
        for att in att_de:
            f.write("{}, ".format(att))
        f.write("\n")
    print("save att to {}".format(csv_path))    

def extract_exp_test_att(filepath_prefix, seeds=[6, 66, 666]):
    """
    解析单个实验中多个地图的结果，提取 att 值，保存到 csv 文件
    filepath: 实验结果路径. 如：records/att_hz_8_
    """
    # 获取指定目录下的所有文件和子目录
    for seed in seeds:
        filepath = filepath_prefix + str(seed)
        for item in os.listdir(filepath):
            extract_test_att(os.path.join(filepath, item))

def merge_exp_att_csv_plot(filepath_prefix, seeds=[6, 66, 666], max_episode = 80):
    """
    合并多个实验结果的 att.csv 文件，针对统一参数不同seed的实验结果
    filepath_prefix: 实验结果路径. 如：records/att_hz_8_
    """
    map_path = {}
    for seed in seeds:
        filepath = filepath_prefix + str(seed)
        for item in os.listdir(filepath):
            tmp_path = os.path.join(filepath, item)
            if not os.path.isdir(tmp_path):
                continue
            map_name = item.split(".")[0]
            if map_name not in map_path.keys():
                map_path[map_name] = []
            map_path[map_name].append(os.path.join(filepath, item, "att.csv"))
    print(f'find {len(map_path)} maps in {filepath_prefix}')
    # print(map_path)

    for map_name, csv_list in map_path.items():
        # print(map_name, csv_list)
        print(f"merge {map_name}, len={len(csv_list)}")
        csv_data = {}   # 存储一个map,不同seed的结果
        att_list = None
        for csv_path in csv_list:
            # print(f"read {csv_path}")
            with open(csv_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    att_list = list(map(float,  line.split(", ")[:-1]))
            if len(att_list) != 0:
                # print(f"read {csv_path}, len={len(att_list)} att_list={att_list}")
                seed = int(csv_path.split("/")[1].split("_")[-1])
                csv_data[seed] = att_list
        # print(csv_data)
                
        # save csv_data to csv
        plt_data = {
            "min_att": deque(maxlen=max_episode),
            "max_att": deque(maxlen=max_episode),
            "mean_att": deque(maxlen=max_episode),
        }
        # for csv_path in csv_list:
        #     # 保存路径不一样，但内容是一样的
        csv_path = csv_list[0]
        targetpath = csv_path.split("/")[:2]
        targetpath = os.path.join(*targetpath, "{}_merged_att.csv".format(map_name))
        att_de = deque(maxlen=len(seeds))
        with open(targetpath, "w") as f:
            # 写入数据表的表头
            f.write("step, ")
            for seed in seeds:
                f.write("{}, ".format(seed))
            f.write("min_att, max_att, mean_att\n") # 额外添加三个指标
            # 写入数据内容
            for i in range(max_episode):
                f.write("{}, ".format(i+1))
                for seed in seeds:
                    f.write("{}, ".format(csv_data[seed][i]))
                    att_de.append(csv_data[seed][i])
                min_att = min(att_de)
                max_att = max(att_de)
                mean_att = round(sum(att_de)/len(seeds), 2)
                plt_data["min_att"].append(min_att)
                plt_data["max_att"].append(max_att)
                plt_data["mean_att"].append(mean_att)
                f.write("{}, {}, {}\n".format(min_att, max_att, mean_att))
        print(f"save to {targetpath}")

        
        # # 绘制曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, 81)), plt_data["mean_att"], color='blue')  # 绘制平均值曲线
        plt.fill_between(list(range(1, 81)), plt_data["min_att"], plt_data["max_att"], color='blue', alpha=0.2) # 绘制阴影
        plt.xlabel("step")
        plt.ylabel("att")
        plt.legend()    # 添加图例
        targetpath = csv_path.split("/")[:2]
        targetpath = os.path.join(*targetpath, "{}.png".format(map_name))
        plt.savefig(targetpath)

def merge_plot(filepath_prefix, seeds=[6, 66, 666]):
    """
    针对同一参数不同seed的实验结果，汇总结果+绘图。结果文件保存在第一个seed的目录下
    """
    for seed in seeds:
        filepath = filepath_prefix + str(seed)
        for item in os.listdir(filepath):
            tmp_path = os.path.join(filepath, item)
            if not os.path.isdir(tmp_path):
                continue
            extract_test_att(tmp_path)
    
    extract_exp_test_att(filepath_prefix,seeds)

    merge_exp_att_csv_plot(filepath_prefix, seeds)

if __name__ == "__main__":
    """Only use these data"""
    total_summary = {
        "traffic": [],
        "final_duration": [],
        "final_duration_std": [],
        "final_through": [],
    }
    memo = "att_ny_8_666"
    # summary_detail_RL(memo, copy.deepcopy(total_summary))
    # summary_detail_conventional(memo)
    get_att_eng(memo)

    filepath_prefix = "records/att_ny_8_"
    merge_plot(filepath_prefix)
