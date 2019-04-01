import argparse
import torch
import numpy as np
import torch.utils.data as data
import os
from torch.autograd import Variable
from torch.nn import CosineSimilarity
from deep_conv_sv import SpeakerVerification
from preprocess import readEnrollmentPaths, readTestPaths
import sidekit
import tqdm
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def calculateOneEmbedding(model, eval_feature_path, path):
    server = sidekit.FeaturesServer(features_extractor=None,
                                    feature_filename_structure=eval_feature_path+"/{}.h5",
                                    sources=None,
                                    dataset_list=["energy","cep","vad"],
                                    mask="[0-19]",
                                    feat_norm="cmvn",
                                    global_cmvn=None,
                                    dct_pca=False,
                                    dct_pca_config=None,
                                    sdc=False,
                                    sdc_config=None,
                                    delta=True,
                                    double_delta=False,
                                    delta_filter=None,
                                    context=None,
                                    traps_dct_nb=None,
                                    rasta=True,
                                    keep_all_features=False)                                  
    feature, _ = server.load(path, channel = 0)
    feature = feature.astype(np.float32)
    feature = torch.tensor(feature).unsqueeze(0)
    feature = feature.unsqueeze_(0)
    embedding, _ = model(feature)
    embedding = embedding / embedding.pow(2).sum(dim = 1).sqrt()
    return embedding # calculate the embedding

def enrollment(model, eval_path, enroll_path):
    # spk_indices是说话人-文件列表，grp_indices是组编号-说话人集合，classes说话人列表（有重复）
    spk_indices, grp_indices, _ = readEnrollmentPaths(enroll_path)
    spk_embeddings = {} #说话人-注册embedding 30个
    grp_embeddings = {} #组编号-组内注册embeddings 6个，每组5个embeddings
    for key in grp_indices.keys():
        grp_embeddings[key] = []
    for key, value in spk_indices.items():
        num_utt = len(value)
        spk_embeddings[key] = torch.zeros((1, 512))
        for v in value:
            spk_embedding = calculateOneEmbedding(model, eval_path, v) # spk_embedding
            spk_embeddings[key] += spk_embedding
        spk_embeddings[key] /= num_utt
    for key, value in spk_embeddings.items(): # 说话人-embedding列表 30-30
        for k, v in grp_indices.items():  # 组编号-说话人set 6-30
            if key in v:
                if not type(grp_embeddings[k]) == list:
                    grp_embeddings[k] = torch.cat((grp_embeddings[k], value), dim = 0)
                else:
                    grp_embeddings[k] = value
    return spk_embeddings, grp_embeddings

def evaluate(model, eval_feature_path, enrollment_path, eval_path, annotation_path):
    model.eval()
    _, grp_embeddings = enrollment(model, eval_feature_path, enrollment_path)
    indices, _ = readTestPaths(eval_path) # 组编号-文件名列表
    cosine_similarity = CosineSimilarity(dim = 1)
    cos_similarity = {}
    for key, value in indices.items(): # 组编号-文件名列表
        for path in value:          
            out = calculateOneEmbedding(model, eval_feature_path, path) # test embedding
            cosine = cosine_similarity(out, grp_embeddings[key]) # grp_embeddings[key] is the corresponding enroll embeddings
            cos_similarity[path] = max(cosine).item() # 距离越远越好
    accuracy, threshold = acc(cos_similarity, annotation_path)
    print('ACCURACY: ', accuracy)
    return accuracy, threshold

def acc(cos_similarity, annotation_path):
    data = pd.read_csv(annotation_path)
    maximum = max(cos_similarity.values())
    minimum = min(cos_similarity.values())
    result = []
    threshold = []
    for i in tqdm.tqdm(np.linspace(minimum, maximum, 300)):
        threshold.append(i)
        correct = 0
        for idx, j in enumerate(data['FileID']):
            if cos_similarity[j] > i and data['IsMember'][idx] == 'Y':
                correct += 1
            if cos_similarity[j] < i and data['IsMember'][idx] == 'N':
                correct += 1
        accuracy = correct / len(data['FileID'])
        result.append(accuracy)
    result = np.array(result)
    idx = np.argmax(result)
    return result[idx], threshold[idx]


def predict(model, test_feature_path, enrollment_path, test_path, threshold):
    model.cpu()
    _, grp_embeddings = enrollment(model, test_feature_path, enrollment_path)
    indices, _ = readTestPaths(test_path)  # 组编号-文件名列表
    cosine_similarity = CosineSimilarity(dim = 1)
    cos_similarity = {}
    for key, value in indices.items():  # 组编号-文件名列表
        for path in value:
            out = calculateOneEmbedding(model, test_feature_path, path)
            cosine = cosine_similarity(out, grp_embeddings[key])
            cos_similarity[path] = max(cosine).item()
    groupid = []
    fileid = []
    ismember = []
    results = {}
    for idx, (k, v) in enumerate(cos_similarity.items()):
        groupid.append(idx // 100)
        fileid.append(k)
        ismember.append('Y') if v > threshold else ismember.append('N')
    results['GroupID'] = groupid
    results['FileID'] = fileid
    results['IsMember'] = ismember
    results = pd.DataFrame(results)
    results.to_csv('results.csv', index=False)