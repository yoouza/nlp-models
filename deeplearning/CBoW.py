from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string
from tqdm import tqdm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CBoW:
    def __init__(self, doc, N_dim, N_window, alpha, epoch):
        self.doc = doc
        self.N_dim = N_dim
        self.N_window = N_window
        self.alpha = alpha
        self.epoch = epoch

    def _tokenize(self):
        pos_unused = ['CC', 'DT']
        stopwords = list(string.punctuation)
        words = pos_tag(word_tokenize(self.doc))
        words = [w[0] for w in words if w[1] not in pos_unused and w[0] not in stopwords]
        token = np.unique(words)
        self.token = token
        self.words = words
        return words, token

    def _init_weight(self):
        W1 = np.random.random_sample((len(self.token), self.N_dim))
        W2 = np.random.random_sample((self.N_dim, len(self.token)))
        return W1, W2

    def _onehot_encoding(self):
        one_hot = {}
        for i, word in enumerate(self.token):
            one_hot[word] = np.eye(len(self.token))[i]
        return one_hot
    
    def _softmax(self, v):
        return np.exp(v) / sum(np.exp(v))

    def optimize(self):
        words, token = self._tokenize()
        one_hot = self._onehot_encoding()
        W1, W2 = self._init_weight()
        losses_epoch = []

        for e in tqdm(range(self.epoch)):
            losses = []
            for i, word in enumerate(words):
                output_vec = one_hot[word]
                input_vec = []

                for j in range(1, self.N_window+1):
                    if i-j >= 0:
                        input_vec.append(one_hot[words[i-j]]) # 전 단어
                    try: input_vec.append(one_hot[words[i+j]]) # 후 단어
                    except: pass
                
                H = []
                for j in range(len(input_vec)):
                    H.append(np.dot(input_vec[j], W1))
                H = np.sum(H, axis=0) / len(input_vec) # 모든 window 속 단어들의 평균값

                pred = self._softmax(np.dot(H, W2))

                loss = -np.dot(output_vec, np.log(pred).reshape(len(pred), 1))
                losses.extend(loss)
                
                gradient_W2 = np.dot(H.reshape(len(H), 1), (pred-output_vec).reshape(1, len(pred)))
                gradient_W1 = np.dot(input_vec[j].reshape(len(input_vec[j]), 1), np.dot(W2, (pred-output_vec)).reshape(1, len(W2)))
                
                W2 -= self.alpha * gradient_W2
                W1 -= self.alpha * gradient_W1
            
            losses_epoch.append(np.sum(losses))
        
        self.W1 = W1
        self.W2 = W2
        self.losses_epoch = losses_epoch

    def plot(self):    
        plt.plot(np.arange(self.epoch), self.losses_epoch)
        plt.title("Loss of CBoW", size=15)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.show()
    
    def similarity(self):
        return pd.DataFrame(self.W1, index = self.token)
    
    def similar_tokens(self, N_similar):
        similarity_minus = {}
        similarity_df = {}

        for i in range(len(self.token)):
            tmp = []
            for j in range(len(self.token)):
                similarity = np.dot(self.W1[i], self.W1[j]) / np.linalg.norm(self.W1[i]) * np.linalg.norm(self.W1[j])
                tmp.append(-similarity) # 순서 정렬을 위해 음수 처리
            similarity_minus[self.token[i]] = tmp

        for i, t in enumerate(self.token, N_similar):
            tmp = np.argsort(similarity_minus[t]) # 순서대로 유사도가 높은 것
            similarity_df[t] = self.token[np.delete(tmp, np.where(tmp == i))][:N_similar] # 같은 단어 출력 방지
        
        return pd.DataFrame(similarity_df)

    def accuracy(self):
        acc = 0
        for i in range(len(self.words)-3):
            word_1 = np.where(self.token == self.words[i])
            word_2 = np.where(self.token == self.words[i+1])
            word_3 = np.where(self.token == self.words[i+2])
            w = self.W1[word_1] - self.W1[word_2] + self.W1[word_3]

            simmilar_w = []
            for j in range(len(self.token)):
                tmp = np.dot(w, self.W1[j]) / np.linalg.norm(w) * np.linalg.norm(self.W1[j])
                simmilar_w.append(tmp)
            acc += (self.token[np.argmax(simmilar_w)] == self.words[i+3]) * 1

        acc = acc / (len(self.words) - 3)
        return acc


if __name__ == "__main__":
    doc = "you will never know until you try."
    cb = CBoW(doc, 4, 2, 0.01, 10000)
    
    cb.optimize()
    cb.plot()
    cb.similarity()
    cb.similar_tokens(3)
    cb.accuracy()