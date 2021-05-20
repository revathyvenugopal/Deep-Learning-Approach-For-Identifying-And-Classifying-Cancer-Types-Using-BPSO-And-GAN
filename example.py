import binary_optimization as opt
import numpy as np
from sklearn import svm
from sklearn import model_selection as ms

#import pandas as pd
# df=pd.read_csv(r"C:\Users\sona jose\Downloads\labels_0.csv")
# print (df.info)


with open(r"/home/ubuntu/Downloads/labels_0.csv") as f:
    tr_l = np.array([[float(d) for d in data.split(',')] for data in f.read().splitlines()])
with open(r"/home/ubuntu/Downloads/data_0.csv") as f:
    tr_d = np.array([[float(d) for d in data.split(',')] for data in f.read().splitlines()])
with open(r"/home/ubuntu/Downloads/labels_1.csv") as f:
    te_l = np.array([[float(d) for d in data.split(',')] for data in f.read().splitlines()])
with open(r"/home/ubuntu/Downloads/data_1.csv") as f:
    te_d = np.array([[float(d) for d in data.split(',')] for data in f.read().splitlines()])

def test_score(gen,tr_x,tr_y,te_x,te_y):
    clf = svm.LinearSVC()
    mask=np.array(gen) == 1
    al_data=np.array(tr_x[:,mask])
    al_test_data=np.array(te_x[:,mask])
    return np.mean([svm.LinearSVC().fit(al_data,tr_y).score(al_test_data,te_y) for i in range(4)])
class Evaluate:
    def __init__(self):
        self.train_l = tr_l
        self.train_d = tr_d
        self.K = 4
    def evaluate(self,gen):
        mask=np.array(gen) > 0
        al_data=np.array([al[mask] for al in self.train_d])
        print(al_data)
        kf = ms.KFold(n_splits=self.K)
        print(kf)
        s = 0
        for tr_ix,te_ix in kf.split(al_data):
            s+= svm.LinearSVC().fit(al_data[tr_ix],self.train_l[tr_ix].ravel()).score(al_data[te_ix],self.train_l[te_ix])
        s/=self.K
        print(s)
        return s


    def check_dimentions(self,dim):
        if dim==None:
            return len(self.train_d[0])
        else:
            return dim
s,g,l=opt.BPSO(Eval_Func=Evaluate,n=20,m_i=200)
print("BPSO:\n\t{0}   {1:.6f}  {2}  {3:.6f}".format("".join(map(str,g)),s,l,
                                      test_score(g,tr_d,tr_l,te_d,te_l)))