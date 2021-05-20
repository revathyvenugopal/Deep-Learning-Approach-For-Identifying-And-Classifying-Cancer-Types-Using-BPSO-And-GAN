from multiprocessing import Pool
import numpy as np
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm

"""Using Algorithm
* Binary Genetic Algorithm
* Binary Particle Swarm optimization
* Binary Cuckoo Search
* Binary Firefly algorithm
* Binary Bat Algorithm
* Binary Gravitational Search algorithm
* Binary Dragon Fly Algorithm
"""

"""Evaluate Function """
class Evaluate:
    def __init__(self):
        None
    def evaluate(self,gen):
        None
    def check_dimentions(self,dim):
        None

"""Common Function"""
def random_search(n,dim):
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens



def logsig(n): return 1 / (1 + math.exp(-n))
def sign(x): return 1 if x > 0 else (-1 if x!=0 else 0)

def BPSO(Eval_Func,n=20,m_i=200,minf=0,dim=None,prog=False,w1=0.5,c1=1,c2=1,vmax=4):
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    gens=random_search(n,dim)
    pbest=float("-inf") if minf == 0 else float("inf")
    gbest=float("-inf") if minf == 0 else float("inf")
    #vec=3
    #flag=dr
    gens=random_search(n,dim)
    vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    one_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    zero_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]

    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    pbest=dc(fit)
    xpbest=dc(gens)
    #w1=0.5
    if minf==0:
        gbest=max(fit)
        xgbest=gens[fit.index(max(fit))]
    else:
        gbest=min(fit)
        xgbest=gens[fit.index(min(fit))]

    #c1,c2=1,1
    #vmax=4
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        #w=0.5
        for i in range(n):
            if tuple(gens[i]) in gens_dict:
                score=gens_dict[tuple(gens[i])]
            else:
                score=estimate(gens[i])
                gens_dict[tuple(gens[i])]=score
            fit[i]=score
            if fit[i]>pbest[i] if minf==0 else fit[i]<pbest[i]:#max
                pbest[i]=dc(fit[i])
                xpbest[i]=dc(gens[i])

        if minf==0:
            gg=max(fit)
            xgg=gens[fit.index(max(fit))]
        else:
            gg=min(fit)
            xgg=gens[fit.index(min(fit))]

        if gbest<gg if minf==0 else gbest>gg:#max
            gbest=dc(gg)
            xgbest=dc(xgg)

        oneadd=[[0 for d in range(dim)] for i in range(n)]
        zeroadd=[[0 for d in range(dim)] for i in range(n)]
        c3=c1*random.random()
        dd3=c2*random.random()
        for i in range(n):
            for j in range(dim):
                if xpbest[i][j]==0:
                    oneadd[i][j]=oneadd[i][j]-c3
                    zeroadd[i][j]=zeroadd[i][j]+c3
                else:
                    oneadd[i][j]=oneadd[i][j]+c3
                    zeroadd[i][j]=zeroadd[i][j]-c3

                if xgbest[j]==0:
                    oneadd[i][j]=oneadd[i][j]-dd3
                    zeroadd[i][j]=zeroadd[i][j]+dd3
                else:
                    oneadd[i][j]=oneadd[i][j]+dd3
                    zeroadd[i][j]=zeroadd[i][j]-dd3

        one_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(one_vel,oneadd)]
        zero_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(zero_vel,zeroadd)]
        for i in range(n):
            for j in range(dim):
                if abs(vel[i][j]) > vmax:
                    zero_vel[i][j]=vmax*sign(zero_vel[i][j])
                    one_vel[i][j]=vmax*sign(one_vel[i][j])
        for i in range(n):
            for j in range(dim):
                if gens[i][j]==1:
                    vel[i][j]=zero_vel[i][j]
                else:
                    vel[i][j]=one_vel[i][j]
        veln=[[logsig(s[_s]) for _s in range(len(s))] for s in vel]
        temp=[[random.random() for d in range(dim)] for _n in range(n)]
        for i in range(n):
            for j in range(dim):
                if temp[i][j]<veln[i][j]:
                    gens[i][j]= 0 if gens[i][j] ==1 else 1
                else:
                    pass
    return gbest,xgbest,xgbest.count(1)