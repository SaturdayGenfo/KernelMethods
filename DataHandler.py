import pandas as pd
from collections import deque
from tqdm import tqdm

class DataHandler():
    
    def __init__(self, fname):
        self.X = pd.read_csv(fname)['seq']
        self.data = self.X
        self.kmer_set = {}
        self.neigborhoods = {}
        
        self.alph = "GATC"
        self.precomputed = {}
        
    def spectrum_preprocess(self, k):
        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Computing kmer embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer in embedding[i]:
                    embedding[i][kmer] += 1
                else:
                    embedding[i][kmer] = 1
        self.data = embedding
        
        
    def populate_kmer_set(self, k):
        d = len(self.X[0])
        idx = 0
        print("Populating kmer set")
        for x in tqdm(self.X):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.kmer_set:
                    self.kmer_set[kmer] = idx
                    idx +=1  
            
    def mismatch_preprocess(self, k, m):
        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Computing mismatch embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.precomputed:
                    Mneighborhood = self.m_neighborhood(kmer, m)
                    self.precomputed[kmer] = [self.kmer_set[neighbor] for neighbor in Mneighborhood if neighbor in self.kmer_set]
                    
                for idx in self.precomputed[kmer]:
                    if idx in embedding[i]:
                        embedding[i][idx] += 1
                    else:
                        embedding[i][idx] = 1
        self.data = embedding
            
    def m_neighborhood(self, kmer, m):
        mismatch_list = deque([(0, "")])
        for letter in kmer:
            num_candidates = len(mismatch_list)
            for i in range(num_candidates):
                mismatches, candidate = mismatch_list.popleft()
                if mismatches < m :
                    for a in self.alph:
                        if a == letter :
                            mismatch_list.append((mismatches, candidate + a))
                        else:
                            mismatch_list.append((mismatches + 1, candidate + a))
                if mismatches == m:
                    mismatch_list.append((mismatches, candidate + letter))
        return [candidate for mismatches, candidate in mismatch_list]
                
        
