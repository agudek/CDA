from numpy import *
import pandas as pd
import numpy as np
import math

from sklearn.metrics import confusion_matrix

df = pd.read_csv('scenario_10_discretised.csv')
df['StartTime'] = df[['date', 'time']].apply(lambda x: ' '.join(x), axis=1)

# parse the StartTime as datatime
df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['packets']= df['packets'].astype(int)
df['bytes']= df['bytes'].astype(int)

df['packets']=df['packets'].fillna(0)
df['bytes']=df['bytes'].fillna(0)
df['duration']=df['duration'].fillna(0)

def map_attributes(x, split_list):
    for i,s in enumerate(split_list):
        if x > s:
            continue
        return i
    return len(split_list)

def encode_netflow(netflow, fs):
    code = 0
    space_size = fs[0]*fs[1]
    for i in range (0,len(fs)):
        code = code + (netflow[i]) * space_size / fs[i]
        space_size = space_size / fs[i]
    return code

#Ordinal rank implementation from paper 4 (Pellegrino Paper)
def ordinal_rank(bins,column):
    percentile = int(100/bins)
    split_list = []

    for p in range(percentile, 99, percentile):
        rank = math.ceil((p/100.0)*len(df[column])*1.0)
        val = sorted(df[column])[int(rank)]
        split_list.append(val)
    return split_list

split_list_Packets = ordinal_rank(4, 'packets')

window_size = 2
infected_host = "147.32.84.165"

data_infected = df.loc[ (df['source_ip'] == infected_host) | (df['destination_ip'] == infected_host)]

d_discr_infected = pd.DataFrame()
d_discr_infected['packets'] = data_infected['packets'].apply(lambda x: map_attributes(x,split_list_Packets))
d_discr_infected['protocol'] = pd.factorize(data_infected['protocol'])[0]
feature_space = [d_discr_infected[name].nunique() for name in d_discr_infected.columns[0:2]]
d_discr_infected['code'] = d_discr_infected.apply(lambda x: encode_netflow(x, feature_space),axis=1)


d_discr = pd.DataFrame()
d_discr['packets'] = df['packets'].apply(lambda x: map_attributes(x,split_list_Packets))
d_discr['protocol'] = pd.factorize(df['protocol'])[0]
feature_space = [d_discr[name].nunique() for name in d_discr.columns[0:2]]
d_discr['code'] = d_discr.apply(lambda x: encode_netflow(x, feature_space),axis=1)
df['code'] = d_discr['code']


d_discr['source_ip'] = df['source_ip']
d_discr['destination_ip'] = df['destination_ip']
d_discr['StartTime'] = df['StartTime']


def extract_state(host_data, width):
    start_time = host_data['StartTime']
    difference_list = []
    for i in range(len(host_data)):
        if i == 0:
            difference = 0
        else:
            difference = start_time.iloc[i] - start_time.iloc[i - 1]
            difference = np.ceil(difference.value / 1e6)
        difference_list.append(difference)
    host_data['time'] = difference_list

    # keep the hosts in the specified sliding window
    state_list = []
    for i in range(len(host_data)):
        j = i
        state_list.append([])
        temp_list = [host_data['code'].iloc[j]]
        time_sum = 0
        while True:
            try:
                time_sum += difference_list[j + 1]
            except:
                break
            j += 1
            if time_sum <= width:
                temp_list.append(host_data['code'].iloc[j])
            else:
                break
        if len(temp_list) >= 3:
            state_list[i] = temp_list
    host_data['window_states'] = state_list
    return host_data


def ngram(states, n):
    ngrams = []
    for state in states:
        for s in range(len(state)-n+1):
            ngrams.append(state[s:s+n])
    return ngrams


def combine_ngrams(a,b):
    maxc = 1
    for sfrom in b:
        if sfrom in a:
            for sto in b[sfrom]:
                if sto in a[sfrom]:
                    (c,x) = a[sfrom][sto]
                    (c2, x2) = b[sfrom][sto]
                    a[sfrom][sto] = (c+c2, x+x2)
                else:
                    a[sfrom][sto] = b[sfrom][sto]
        else:
            a[sfrom] = b[sfrom]

def sort_ngrams(grams3_normals):
    ngram_dict = {}
    for gram in grams3_normals :
        grams = str(gram)[1:-1]
        if grams in ngram_dict:
            ngram_dict[grams] += 1
        else:
            ngram_dict[grams] = 1
    sorted_ngrams = sorted(ngram_dict.items(),key = lambda x:x[1], reverse = True )
    sortedgrams_normed = [ (list[0], 1.0*list[1]/len(grams3_normals)) for list in sorted_ngrams]
    return sortedgrams_normed

def fingerprint_matching(finger_train, finger_test, topN):
    finger_train = finger_train[0:topN]
    freq_train = [pair[1] for pair in finger_train]

    finger_test = {pair[0]: pair[1] for pair in finger_test}
    fre_test = []
    for i in range(topN):
        key = finger_train[i][0]
        if key in finger_test:
            fre_test.append(finger_test[key])
        else:
            fre_test.append(0)
    dis = distance(freq_train, fre_test)
    return dis

def distance(finger_train, finger_test):
    finger_train = np.array(finger_train)
    finger_test = np.array(finger_test)
    dis = sum((np.divide((finger_train-finger_test),(finger_train+finger_test)/2))**2)
    return dis

normal_host = '147.32.84.170'

test_hosts = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
             '147.32.84.205','147.32.84.206','147.32.84.207','147.32.84.209', '147.32.84.134',
              '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

train_normal = df[(df['source_ip'] == normal_host) | (df['destination_ip'] == normal_host)]
train_normal_states = extract_state(train_normal, width=90)
train_normal_states = [l for l in train_normal_states['window_states'] if len(l)>0]
train_normal_ngrams = ngram(train_normal_states, window_size)
train_normal = sort_ngrams(train_normal_ngrams)


# fingerprinting the infected host used as train
train_infected = d_discr[(d_discr['source_ip'] == infected_host) | (d_discr['destination_ip'] == infected_host)]
train_infected_states = extract_state(train_infected, width=90)
train_infected_states = [l for l in train_infected_states['window_states'] if len(l)>0]
train_infected_ngrams = ngram(train_infected_states, window_size)
train_infected = sort_ngrams(train_infected_ngrams)

fmatch_test= np.zeros((len(test_hosts),window_size))

for index, host in enumerate(test_hosts):
    test_data = d_discr[(d_discr['source_ip'] == host) | (d_discr['destination_ip'] == host)]
    test_states = extract_state(test_data,width=90)
    test_states = [l for l in test_states['window_states'] if len(l)>0]
    test_ngrams = ngram(test_states, window_size)
    test_fingerprint = sort_ngrams(test_ngrams)
    fmatch_test[index][0] = fingerprint_matching(train_infected,test_fingerprint,9)
    fmatch_test[index][1] = fingerprint_matching(train_normal,test_fingerprint,9)

test_label = np.zeros(14)

for i in range(14):
        if fmatch_test[i][0] <= fmatch_test[i][1]:
            test_label[i] = 1
        else:
            test_label[i] = 0

true_label = [1,1,1,1,1,1,1,1,1,0,0,0,0,0]

tn, fp, fn, tp = confusion_matrix(test_label, true_label).ravel()

print("true positives:", tp)
print("false positives:", fp)
print("false Negative:", fn)
print("true negatives:", tn)

print ('precision:', float(tp)/(tp+fp))
print ('recall', float(tp)/(tp+fn))