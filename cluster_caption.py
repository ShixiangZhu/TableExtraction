import os
import json
import fastText
import numpy as np
from sklearn.cluster import KMeans  
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_json(directory):
	"""
	input: dir
	output: dictionary of paper #: list of captions
	"""
	dict_paper_table = {}
	index_paper = 1
	dict_list_index  = {}
	for filename in os.listdir(directory):
		dict_list_index[ "p" + str(index_paper)] = filename
		key = "p" + str(index_paper) + ", " + filename 
		with open(directory + "/" + filename, 'r') as f:
			list_table = json.load(f)
			list_caption = []
			for data in list_table:
				if data["figType"] == "Table":
					list_caption.append(data["caption"])
			if(len(list_caption) > 0):
				dict_paper_table[key] = list_caption
		index_paper += 1
	return dict_paper_table, dict_list_index

def get_sentence_vector(dict_paper_table):
	"""
	input: dictionary of paper: table captions
	output: 
	"""
	ftmodel = fastText.load_model('../wiki.en.bin')
	dict_key_caption = {} # key = p1t1  ==> paper's name, caption 
	key_list = []
	vec_list = []
	caption_list = []
	for key, value in dict_paper_table.items():
		keylist = key.split(",")
		for i in range(len(value)):
			sen = value[i]
			newKey = keylist[0] + "t" + str(i)
			newValue = keylist[1]  + "  ==>  " + sen
			dict_key_caption[newKey]  =  newValue
			key_list.append(newKey)
			vec_list.append(ftmodel.get_sentence_vector(sen.lower().strip()))
			caption_list.append(sen)
	return key_list, np.asarray(vec_list), dict_key_caption


def k_means_cluster(vec_list):
	"""
	input: vector
	return: label
	"""
	random.seed(1)
	kmeans = KMeans(n_clusters=10, max_iter = 10000)   #n_clusters:number of cluster  
	kmeans.fit(vec_list)  
	model = TSNE(learning_rate = 90, n_components = 2)
	tsne_visualize = model.fit_transform(vec_list)
	return kmeans.labels_, tsne_visualize


def plot2D(list_paper, list_tsne, list_kmeans):
    print(list_paper)
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    dict_k = {}
    for i in range(len(list_kmeans)):
        if list_kmeans[i] in dict_k:
            dict_k[list_kmeans[i]].append(list_tsne[i])
        else:
            temp_list = []
            temp_list.append(list_tsne[i])
            dict_k[list_kmeans[i]] = temp_list
    for key, value in dict_k.items():
        x = []
        y = []
        for pos in value:
            x.append(pos[0])
            y.append(pos[1])
        plt.scatter(x, y, color = colors[key])
    for i, txt in enumerate(list_paper):
        plt.annotate(txt, (list_tsne[i][0], list_tsne[i][1]))
    plt.show()

def plot3D(list_paper, list_tsne, list_kmeans):
    print(list_paper)
    colors = ['#000000', '#911eb4', '#ffe119', '#f032e6', '#911eb4', '#800000', '#808000', '#f58231', '#fabebe',  '#42d4f4']
    dict_k = {}
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(list_kmeans)):
        if list_kmeans[i] in dict_k:
            dict_k[list_kmeans[i]].append(list_tsne[i])
        else:
            temp_list = []
            temp_list.append(list_tsne[i])
            dict_k[list_kmeans[i]] = temp_list
    for key, value in dict_k.items():
        x = []
        y = []
        z = []
        for pos in value:
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        ax.scatter(x, y, z, color = colors[key])

def main():
	# extract paper information
	root_dir =  "/Users/zhengshuangjing/desktop/json_data"
	dict_paper_table, dict_list_index = read_json(root_dir)  # dict_list_index is ==> p1 ==> paper  name
	
	# get caption feature
	key_list, vec_list, dict_key_caption = get_sentence_vector(dict_paper_table) # 这点的  order 一样
	
	# run baseline clustering
	label_kmeans, tsne_visualize = k_means_cluster(vec_list)

	# visulazation
	plot2D(key_list, tsne_visualize, label_kmeans)


if __name__ == '__main__':
  main()