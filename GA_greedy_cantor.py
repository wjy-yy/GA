import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import itertools

point = []


def dis(x, y):
    d = np.array(x) - np.array(y)
    return np.sqrt(np.sum(d**2))


def floyd(p):
    f = []
    for i in p:
        f.append([])
    for idx, x in enumerate(p):
        for y in p:
            f[idx].append(dis(x,y))
    for kdx, x in enumerate(p):
        for idx, y in enumerate(f):
            for jdx, z in enumerate(y):
                if f[idx][kdx] > f[idx][jdx]+f[jdx][kdx]:
                    f[idx][kdx] = f[idx][jdx] + f[jdx][kdx]
    return f

def drawPic(dots):
    # plt.figure(figsize=(10,6))
    plt.xlim(0,30,0.001)     #x轴的刻度范围
    plt.ylim(0,30,0.001)       #y轴的刻度范围
    plt.xlabel('x')    #x轴的标题
    plt.ylabel('y')    #y轴的标题
	#绘制各个点及点所代表地点名称
    for i in range(len(dots)-1):
        # plt.text(dots[i][0],dots[i][1],color='#0085c3')
        plt.plot(dots[i][0],dots[i][1],'o',color='#0085c3')
    #连接各个点
    for i in range(-1,len(dots)-1):
        start = (dots[i][0],dots[i+1][0])
        end = (dots[i][1],dots[i+1][1])
        plt.plot(start,end,color='#0085c3')
    plt.savefig('Cantor Greedy Gra ' + 'pc=' + str(pc) + ' pm=' + str(pm) + ' dis=' + str(loss[-1]) + '.pdf', dpi=600, format='pdf')
    plt.show()

fac = []

def cantor(a):
    ans = 1
    for idx, i in enumerate(a):
        cnt = 0
        for j in range(idx+1,len(a)):
            if a[j]<i:
                cnt += 1
        ans += cnt * fac[len(a)-idx-1]
    return ans

def incantor(x, len):
    if x>fac[len]:
        return [i for i in range(1,len+1)]
    p = []
    used = [0 for i in range(len+1)]
    x -= 1
    for i in range(len):
        t = x//fac[len-i-1]+1
        for j in range(1,len+1):
            if used[j] == 0:
                t-=1
            if t==0:
                p.append(j)
                used[j]=1
                break
        x%=fac[len-i-1]
    return p
# def main():
if __name__ == '__main__':
    fac.append(1)
    for i in range(1, 101):
        fac.append(i*fac[-1])
    tm = time.time()
    point.append([0, 0])
    point.append([1, 3])
    point.append([3, 4])
    point.append([5, 2])
    point.append([3, 3])
    point.append([11, 8])
    point.append([27, 17])
    point.append([16, 4])
    point.append([14, 21])
    point.append([3, 24])
    point.append([21, 9])
    point.append([14, 19])
    point.append([9, 7])
    point.append([13, 2])
    point.append([3, 23])
    point.append([17, 14])
    point.append([12, 12])
    point.append([4, 17])
    point.append([9, 12])
    point.append([5, 20])

    point.append([25, 25])
    point.append([27, 28])
    point.append([10, 19])
    point.append([11, 29])
    point.append([7, 16])
    point.append([25, 8])
    point.append([27, 1])
    point.append([20, 22])
    point.append([1, 10])
    point.append([13, 4])
    point.append([29, 9])
    point.append([28, 17])
    point.append([10, 29])
    point.append([22, 25])
    point.append([12, 27])
    point.append([2, 24])
    point.append([19, 17])
    point.append([21, 24])
    point.append([22, 6])
    point.append([3, 29])

    point.append([16, 26])
    point.append([23, 20])
    point.append([5, 8])
    point.append([10, 3])
    point.append([29, 7])
    point.append([15, 24])
    point.append([20, 15])
    point.append([22, 23])
    point.append([12, 2])
    point.append([3, 9])
    permu = []
    for i in range(1,len(point)):
        p = []
        p.append(0)
        p.append(i)
        for j in range(2,len(point)):
            mn = 1e9
            nxt = None
            for k in range(len(point)):
                if k not in p:
                    if dis(point[p[-1]],point[k]) < mn:
                        mn=dis(point[p[-1]],point[k])
                        nxt=k
            p.append(nxt)
        permu.append(cantor(np.array(p)+1))
    for i in range(101-len(point)):
        np.random.seed(2*i)
        # p = point
        x = (np.random.randint(1, 2**63-1) * (2**63) +
            np.random.randint(1, 2**63-1)) % fac[len(point)] +1
        permu.append(x)
        # permu[-1].append(permu[-1][0])
    # permu[] is a list of several permutations, representing the choromosomes
    t = 0
    flag = 1
    pc = .8
    pm = .0003
    loss = []
    ans = []
    cont = 0
    # while t < 200 or flag:
    import math
    up = math.ceil(math.log2(fac[len(point)]))
    while t < 1000:
        fit = []
        cro = []
        for i in permu:
            # sumlen = 0
            # for j in range(len(i)):
            #     sumlen += dis(i[j-1],i[j])
            # fit.append(1/sumlen)
            if random.random() < pc:
                cro.append(i)
        # lists in cro should be crossovered
        # print(cro)
        # cont = 0
        random.shuffle(cro)
        for i in range(len(cro)//2):
            # i,i+1 crossover
            pos = 2 ** random.randint(0,up)
            son1 = cro[i]%pos + cro[i+1]//pos*pos
            son2 = cro[i+1]%pos + cro[i]//pos*pos
            # son1 = [[-1,-1] for j in range(len(point))]
            # son2 = [[-1,-1] for j in range(len(point))]
            # for j in range(len(point)):
            #     if random.random() < .5:
            #         son1[j]=cro[i][j]
            #         # son2.append(cro[i+1][j])
            #     else:
            #         son2[j]=cro[i+1][j]
            #         # son2.append(cro[i][j])
            # id = 0
            # for j in cro[i+1]:
            #     while id < len(son1) and son1[id]!=[-1,-1]:
            #         id+=1
            #     if not j in son1:
            #         son1[id]=j
            # id = 0
            # for j in cro[i]:
            #     while id < len(son2) and son2[id]!=[-1,-1]:
            #         id+=1
            #     if not j in son2:
            #         son2[id]=j
            while son1 > fac[len(point)] or son2 > fac[len(point)]:
                pos = 2 ** random.randint(0, up)
                son1 = cro[i] % pos + cro[i + 1] // pos * pos
                son2 = cro[i + 1] % pos + cro[i] // pos * pos
            permu.append(son1)
            permu.append(son2)
        for idx,i in enumerate(permu):
            for j in range(up):
                if random.random() < pm:
                    permu[idx]^=2**random.randint(0,j)
                    # x = random.randint(0,len(point)-1)
                    # y = random.randint(0,len(point)-1)
                    # permu[i][x], permu[i][y] = permu[i][y], permu[i][x]
        for idx, i in enumerate(permu):
            sumlen = 0
            p = np.array(incantor(i,len(point)))
            p -= 1
            for j in range(len(point)):
                sumlen += dis(point[p[j-1]],point[p[j]])
            fit.append([1/sumlen,idx])
        # choose the best 50 chromosomes
        fit.sort(key=lambda x: -x[0])
        dd = []
        for i in range(100,len(fit)):
            dd.append(fit[i][1])
        dd.sort(key=lambda x: -x)
        # print(dd)
        ans = permu[fit[0][1]]
        for i in dd:
            del permu[i]
        # print(len(permu))
        print('epoch:',t,'/2000', 'shortest:',1/fit[0][0])
        loss.append(1/fit[0][0])
        t+=1
        if t>500 and abs(loss[-1] - loss[-2]) < 1e-2:
            cont += 1
        else:
            cont = 0
        # print(cont)
        if t>800 and cont>100:
            break
    print(tm - time.time())
    drawPic([point[i] for i in (np.array(incantor(ans,len(point)))-1)])
    plt.title('Distance changing '+'pc='+str(pc)+' pm='+str(pm))
    plt.xlabel('Iters')  # x轴标签
    plt.ylabel('Dis')  # y轴标签
    plt.plot(loss, linewidth=1, linestyle="solid", label="train loss")
    plt.savefig('Cantor Greedy Dis '+'pc='+str(pc)+' pm='+str(pm)+' dis='+str(loss[-1])+'.pdf', dpi=600, format='pdf')
    plt.show()
