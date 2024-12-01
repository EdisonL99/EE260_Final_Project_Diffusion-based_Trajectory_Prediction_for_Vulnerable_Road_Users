import numpy as np
import os
from bisect import bisect_left

#code for processing the SinD data

def parsed_light_data(datalt):

    lighttime = []
    light1list =[]
    light2list = []
    for line in datalt:

        parts = line.split(',')
        timestamp = float(parts[1])
        light1 = int(parts[2])
        light2 = int(parts[3])
        lighttime.append(timestamp)
        light1list.append(light1)
        light2list.append(light2)
    return lighttime, light1list, light2list

def parse_data(data, datalt):
    tracks = {}
    lighttime, light1list, light2list = parsed_light_data(datalt)
    for line in data:
        parts = line.split(',')
        track_id = int(parts[0][1:])
        frame_id = int(parts[1])
        time_id = float(parts[2])
        pos_x = float(parts[4])
        pos_y = float(parts[5])
        light_index = bisect_left(lighttime, time_id)-1
        light1 = light1list[light_index]
        light2 = light2list[light_index]
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append((frame_id, pos_x, pos_y, light1, light2))
    return tracks

def parse_data_b(data, datalt):
    tracks = {}
    lighttime, light1list, light2list = parsed_light_data(datalt)
    for line in data:
        parts = line.split(',')
        type = str(parts[3])
        if type == 'motorcycle': #change the vehicle object type here
            track_id = int(parts[0])
            frame_id = int(parts[1])
            time_id = float(parts[2])
            pos_x = float(parts[4])
            pos_y = float(parts[5])
            light_index = bisect_left(lighttime, time_id)-1
            light1 = light1list[light_index]
            light2 = light2list[light_index]
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append((frame_id, pos_x, pos_y, light1, light2))
    return tracks

def reshape_track(track_data):
    track_data = sorted(track_data, key=lambda x: x[0])
    if len(track_data) < frame_length:
        return None
    reshaped_data = []
    for i in range(len(track_data) - (frame_length-1)):
        frames = []
        for j in range(i, i + frame_length):
            frames.append([track_data[j][1], track_data[j][2], track_data[j][3], track_data[j][4]])
        reshaped_data.append(frames)
    return np.array(reshaped_data)

def reshape_all_tracks(tracks):
    max_frame = 0
    for track_id, track_data in tracks.items():
        if track_data[len(track_data)-1][0]>max_frame:
            max_frame = track_data[len(track_data)-1][0]
    reshaped_tracks = np.ones([len(tracks), max_frame, frame_length, 4])
    reshaped_tracks[:,:,:,0] = 1.2067
    reshaped_tracks[:,:,:,1] = 1.2255
    count = 0
    for track_id, track_data in tracks.items():
        initial_frame = track_data[0][0]
        final_frame = track_data[len(track_data)-1][0]
        reshaped_data = reshape_track(track_data)
        if reshaped_data is not None:
            reshaped_tracks[count, initial_frame:final_frame-(frame_length-2), :, :] = reshaped_data
            count += 1
    return np.array(reshaped_tracks)

def padarray(size, array):
    for i in range(len(array)):
        original_array = array[0]
        # expand the array
        target_size = (original_array.shape[0], frame_length, size, 4)

        # calculate the size to be expanded for each dimension
        pad_width = [(0, target_size[0] - original_array.shape[0]),
                    (0, target_size[1] - original_array.shape[1]),
                    (0, target_size[2] - original_array.shape[2]),
                    (0, target_size[3] - original_array.shape[3])]  

        # use the numpy.pad function to expand the array and fill the expanded parts with 1
        padded_array = np.pad(original_array, pad_width, mode='constant', constant_values=1.2255)
        array[i] = padded_array
    return array

#length of history and prediction
frame_length = 30
path = "/home/gene/Downloads/SinD"
files = os.listdir(path)

concatenated_array = []
#bycicle
size = 0 # max size of actor num
#adjust the fine[:*] number, because there may not be enough memory to store the data
for file in files[:7]:
    # pedestrian
    data = []
    test_path = path+'/'+file+'/Ped_smoothed_tracks.csv'
    fp = open(test_path, 'r')
    test_list = fp.readlines()
    for test in test_list[1:]:
        test = test.replace('\n','')
        data.append(test)
    #bicyle/tricycle
    # data_b = []
    # test_path_b = path+'/'+file+'/Veh_smoothed_tracks.csv'
    # fp_b = open(test_path_b, 'r')
    # test_list_b = fp_b.readlines()
    # for test in test_list_b[1:]:
    #     test = test.replace('\n','')
    #     data_b.append(test)
    
    datalt = []
    light_path = path+'/'+file+'/TrafficLight_'+file+'.csv'
    fpl = open(light_path, 'r')
    light_list = fpl.readlines()
    for testl in light_list[1:]:
        testl = testl.replace('\n','')
        datalt.append(testl)

    # parse data
    #pedestrian
    parsed_tracks = parse_data(data, datalt)
    #bycicle
    # parsed_tracks = parse_data_b(data_b, datalt)

    # organize the data shape
    reshaped_data = reshape_all_tracks(parsed_tracks)

    arr = np.transpose(reshaped_data, (1, 2, 0, 3))
    if arr.shape[2]>size:
        size = arr.shape[2]
    concatenated_array.append(arr)
    
#enlarge array
concatenated_array = padarray(size, concatenated_array)
arr_transposed = np.concatenate(concatenated_array, axis=0)

# print the results
print("data shape:", reshaped_data.shape)
print("Reshaped data shape:", arr_transposed.shape)
print("mean_x", np.sum(arr_transposed[:,:,:,0])/np.sum(arr_transposed[:,:,:,0]!=0))
print("mean_y", np.sum(arr_transposed[:,:,:,1])/np.sum(arr_transposed[:,:,:,1]!=0))

np.random.shuffle(arr_transposed)
split_index = int(len(arr_transposed)*0.8)
train_data, test_data = arr_transposed[:split_index], arr_transposed[split_index:]
print('trainshape', train_data.shape)
print('testshape', test_data.shape)
# np.save(path+'/SinD_7_train_light_p.npy',train_data)
# np.save(path+'/SinD_7_test_light_p.npy',test_data)
