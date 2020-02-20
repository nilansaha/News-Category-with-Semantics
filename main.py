import math
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from collections import Counter
from joblib import Parallel, delayed

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
stop_words = set(stopwords.words('english'))

newsgroups = fetch_20newsgroups(subset = 'all')

def get_doc_embedding(doc):
  sum_ = np.zeros(300)
  doc = word_tokenize(doc)
  for word in doc:
    if word not in stop_words:
      try:
        sum_ += word2vec_model[word]
      except:
        continue
  return sum_/len(doc)

print('[INFO] Matrix building started')

def build_matrix(start_index, end_index):
  length = len(newsgroups.target)
  if end_index > length:
    end_index = length
  print('Chunk Span {} - {}'.format(start_index, end_index))
  chunk_size = end_index - start_index
  whole_matrix = np.zeros((chunk_size, 300))
  for idx in range(chunk_size):
    whole_matrix[idx] = get_doc_embedding(newsgroups.data[idx + start_index])
  print('[INFO] Part of matrix is built')
  return whole_matrix

chunk_size = math.ceil(len(newsgroups.target)/4)
matrix_chain = Parallel(n_jobs = 8)(delayed(build_matrix)(i, i + chunk_size) for i in range(0, len(newsgroups.target), chunk_size))
whole_matrix = np.concatenate(matrix_chain)

cosine_matrix = squareform(pdist(whole_matrix, 'cosine'))
sorted_matrix = np.argsort(cosine_matrix)
target_matrix = newsgroups.target[sorted_matrix]

def evaluate(k):
  total, correct = 0, 0
  for row in target_matrix:
    actual = row[0]
    predicted = Counter(row[1:1+k]).most_common(1)[0][0]
    total += 1
    if actual == predicted:
      correct += 1
  return correct/total

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
  print(k, evaluate(k))
