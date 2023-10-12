# 预处理
import math

import h5py
import numpy as np
from utils import configs
from sklearn.preprocessing import MinMaxScaler

dtf2 = h5py.File(configs.hdf5_path + "2.hdf5", 'r')
dtf3 = h5py.File(configs.hdf5_path + "3.hdf5", 'r')
dtf4 = h5py.File(configs.hdf5_path + "4.hdf5", 'r')
dtf5 = h5py.File(configs.hdf5_path + "5.hdf5", 'r')
dtf6 = h5py.File(configs.hdf5_path + "6.hdf5", 'r')

def readhdf(file_num, ev_name):
    dataset = []
    if file_num == 2:
        dataset = dtf2.get('data/' + str(ev_name))
    elif file_num == 3:
        dataset = dtf3.get('data/' + str(ev_name))
    elif file_num == 4:
        dataset = dtf4.get('data/' + str(ev_name))
    elif file_num == 5:
        dataset = dtf5.get('data/' + str(ev_name))
    elif file_num == 6:
        dataset = dtf6.get('data/' + str(ev_name))
    return dataset


def readdata(dataPath, args, position_encode):

    dataPath = open(dataPath, 'r', encoding="utf-8")
    data = dataPath.readlines()


    list = []
    for index in range(len(data)):
        dic = {}
        ev_name_station1 = data[index].split(",")[2]
        ev_name_station2 = data[index].split(",")[15]
        ev_name_station3 = data[index].split(",")[28]

        lat = float(data[index].split(",")[5])
        lng = float(data[index].split(",")[6])
        try:
            depth = float(data[index].split(",")[4])
        except:
            print()
        label_lat = [lat]
        label_lng = [lng]
        if depth <= 0:
            depth = 0
        else:
            depth = math.log(depth)
        label_depth = [depth]

        receiver_lat_station1 = float(data[index].split(",")[8])
        receiver_lng_station1 = float(data[index].split(",")[9])
        receiver_lat_station2 = float(data[index].split(",")[21])
        receiver_lng_station2 = float(data[index].split(",")[22])
        receiver_lat_station3 = float(data[index].split(",")[34])
        receiver_lng_station3 = float(data[index].split(",")[35])
        receiver_location_lat = [receiver_lat_station1, receiver_lat_station2, receiver_lat_station3]
        receiver_location_lng = [receiver_lng_station1, receiver_lng_station2, receiver_lng_station3]
        receiver_location = [[receiver_lat_station1, receiver_lng_station1, receiver_lat_station2, receiver_lng_station2,
                              receiver_lat_station3, receiver_lng_station3]]


        ml = [float(data[index].split(",")[3])]

        p_arrive_station1 = np.floor(float(data[index].split(",")[10]))
        p_arrive_station2 = np.floor(float(data[index].split(",")[23]))
        p_arrive_station3 = np.floor(float(data[index].split(",")[36]))

        epicenter_dis1 = (float(data[index].split(",")[12]))
        epicenter_dis2 = (float(data[index].split(",")[25]))
        epicenter_dis3 = (float(data[index].split(",")[38]))

        aiz1 = math.radians(float(data[index].split(",")[-3]))
        aiz2 = math.radians(float(data[index].split(",")[-2]))
        aiz3 = math.radians(float(data[index].split(",")[-1]))

        file_num_station1 = int(data[index].split(",")[0])
        file_num_station2 = int(data[index].split(",")[13])
        file_num_station3 = int(data[index].split(",")[26])

        dataset_station_1 = readhdf(file_num_station1, ev_name_station1)
        dataset_station_2 = readhdf(file_num_station2, ev_name_station2)
        dataset_station_3 = readhdf(file_num_station3, ev_name_station3)

        data_station1 = 0
        data_station2 = 0
        data_station3 = 0
        min_p_arrive_station = 1
        if p_arrive_station1 < p_arrive_station2:
            if p_arrive_station1 < p_arrive_station3:
                min_p_arrive_station = p_arrive_station1
            else:
                min_p_arrive_station = p_arrive_station3
        else:
            if p_arrive_station2 < p_arrive_station3:
                min_p_arrive_station = p_arrive_station2
            else:
                min_p_arrive_station = p_arrive_station3

        receiver_location_lat = [receiver_lat_station1, receiver_lat_station2, receiver_lat_station3]
        receiver_location_lng = [receiver_lng_station1, receiver_lng_station2, receiver_lng_station3]

        if position_encode:
            if int(min_p_arrive_station) + args.seq_len > int(p_arrive_station1):
                data_station1 = np.array(dataset_station_1)[
                                     int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station1 = int(p_arrive_station1) - int(min_p_arrive_station)
            else:
                data_station1 = np.array(dataset_station_1)[
                                int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station1 = args.seq_len + 1
            if int(min_p_arrive_station) + args.seq_len > int(p_arrive_station2):
                data_station2 = np.array(dataset_station_2)[
                                int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station2 = int(p_arrive_station2) - int(min_p_arrive_station)
            else:
                data_station2 = np.array(dataset_station_2)[
                                int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station2 = args.seq_len + 1
            if int(min_p_arrive_station) + args.seq_len > int(p_arrive_station3):
                data_station3 = np.array(dataset_station_3)[
                                int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station3 = int(p_arrive_station3) - int(min_p_arrive_station)
            else:
                data_station3 = np.array(dataset_station_3)[
                                int(min_p_arrive_station):int(min_p_arrive_station) + args.seq_len, :]

                p_arrive_station3 = args.seq_len + 1
        else:
            data_station1 = np.array(dataset_station_1)[
                            int(p_arrive_station1):int(p_arrive_station1) + args.seq_len, :]
            data_station2 = np.array(dataset_station_2)[
                            int(p_arrive_station2):int(p_arrive_station2) + args.seq_len, :]
            data_station3 = np.array(dataset_station_3)[
                            int(p_arrive_station3):int(p_arrive_station3) + args.seq_len, :]
            p_arrive_station1 = 0
            p_arrive_station2 = 0
            p_arrive_station3 = 0


        epicenter_dis = [epicenter_dis1, epicenter_dis2, epicenter_dis3]
        aiz = [aiz1, aiz2, aiz3]

        label_arrivetime = [p_arrive_station1, p_arrive_station2, p_arrive_station3]

        if p_arrive_station1 < p_arrive_station2:
            if p_arrive_station1 < p_arrive_station3:
                if p_arrive_station2 < p_arrive_station3:
                    wavedata = np.concatenate((data_station1, data_station2), axis=1)
                    wavedata = np.concatenate((wavedata, data_station3), axis=1)
                    label_arrivetime = [p_arrive_station1, p_arrive_station2, p_arrive_station3]
                    epicenter_dis = [epicenter_dis1, epicenter_dis2, epicenter_dis3]
                    aiz = [aiz1, aiz2, aiz3]
                else:
                    wavedata = np.concatenate((data_station1, data_station3), axis=1)
                    wavedata = np.concatenate((wavedata, data_station2), axis=1)
                    label_arrivetime = [p_arrive_station1, p_arrive_station3, p_arrive_station2]
                    epicenter_dis = [epicenter_dis1, epicenter_dis3, epicenter_dis2]
                    aiz = [aiz1, aiz3, aiz2]
            else:
                wavedata = np.concatenate((data_station3, data_station1), axis=1)
                wavedata = np.concatenate((wavedata, data_station2), axis=1)
                label_arrivetime = [p_arrive_station3, p_arrive_station1, p_arrive_station2]
                epicenter_dis = [epicenter_dis3, epicenter_dis1, epicenter_dis2]
                aiz = [aiz3, aiz1, aiz2]
        else:
            if p_arrive_station2 < p_arrive_station3:
                if p_arrive_station1 < p_arrive_station3:
                    wavedata = np.concatenate((data_station2, data_station1), axis=1)
                    wavedata = np.concatenate((wavedata, data_station3), axis=1)
                    label_arrivetime = [p_arrive_station2, p_arrive_station1, p_arrive_station3]
                    epicenter_dis = [epicenter_dis2, epicenter_dis1, epicenter_dis3]
                    aiz = [aiz2, aiz1, aiz3]
                else:
                    wavedata = np.concatenate((data_station2, data_station3), axis=1)
                    wavedata = np.concatenate((wavedata, data_station1), axis=1)
                    label_arrivetime = [p_arrive_station2, p_arrive_station3, p_arrive_station1]
                    epicenter_dis = [epicenter_dis2, epicenter_dis3, epicenter_dis1]
                    aiz = [aiz2,aiz3,aiz1]
            else:
                wavedata = np.concatenate((data_station3, data_station2), axis=1)
                wavedata = np.concatenate((wavedata, data_station1), axis=1)
                label_arrivetime = [p_arrive_station3, p_arrive_station2, p_arrive_station1]
                epicenter_dis = [epicenter_dis3, epicenter_dis2, epicenter_dis1]
                aiz = [aiz3,aiz2,aiz1]

        data_y = wavedata[-(args.pred_len + args.label_len):, -args.dec_in:]
        data_y_lat = receiver_location_lat
        data_y_lng = receiver_location_lng
        min_max_scaler = MinMaxScaler()
        label_arrivetime = (np.array(label_arrivetime))
        epicenter_dis = (np.array(epicenter_dis))
        label_depth = np.array(label_depth)
        aiz = np.array(aiz)

        dic['batch_x'] = min_max_scaler.fit_transform(wavedata).T
        dic['batch_y'] = np.array(data_y).T
        dic['batch_y_lat'] = data_y_lat
        dic['batch_y_lng'] = data_y_lng

        dic['label_lat'] = np.array(label_lat)
        dic['label_lng'] = np.array(label_lng)
        dic['label_ml'] = (np.array(ml)-2.2)/1.6
        if np.max(label_arrivetime) == np.min(label_arrivetime):
            dic['label_arrivetime'] = label_arrivetime*0
            dic['label_arrivetime_norm'] = np.array([0, np.min(label_arrivetime)])
        else:
            dic['label_arrivetime'] = (label_arrivetime - np.min(label_arrivetime)) / (
                        np.max(label_arrivetime) - np.min(label_arrivetime))
            dic['label_arrivetime_norm'] = np.array([(np.max(label_arrivetime) - np.min(label_arrivetime)),
                                            np.min(label_arrivetime)])
        if np.max(epicenter_dis) == np.min(epicenter_dis):
            dic['epicenter_dis'] = epicenter_dis*0
            dic['epicenter_dis_norm'] = np.array([0, np.min(epicenter_dis)])
        else:
            dic['epicenter_dis'] = (epicenter_dis-np.min(epicenter_dis))/(np.max(epicenter_dis) - np.min(epicenter_dis))
            dic['epicenter_dis_norm'] = np.array([np.max(epicenter_dis)-np.min(epicenter_dis), np.min(epicenter_dis)])

        if np.max(aiz) == np.min(aiz):
            dic['label_aiz'] = aiz * 0
            dic['label_aiz_norm'] = np.array([0, np.min(aiz)])
        else:
            dic['label_aiz'] = (aiz - np.min(aiz)) / (
                        np.max(aiz) - np.min(aiz))
            dic['label_aiz_norm'] = np.array([np.max(aiz) - np.min(aiz), np.min(aiz)])

        dic['receive_location'] = np.array(receiver_location)
        dic['label_depth'] = ((label_depth))
        dic['label_depth_norm'] = [0]

        list.append(dic)
    return list