# DSNR
Source code of the paper (Version 2):''DSNR: Deep Semantic Network Representation''. Submitted to the ICDM 2020. The project is developed with Python 3.6.4, TenserFlow 1. The develop tool is PyCharm Community Edition 2019.3.1
 
	## Requirement:
 python 3.6.4
 Tensorflow 1.0
 Numpy
 networkx
 Sklearn

Dataset:
Cora, Citeseer and Wiki datasets are given.
including:
.edgelist
.features (node original semantic feature vector representations: attributes of nodes have been processed as TFIDF vector)
.svd (node original semantic feature vector representations: TFIDF matrix on node attributes is decomposed by SVD, and then the left singular matrix obtained is taken as the input feature)
In this paper, we select node TFIDF feature vector
