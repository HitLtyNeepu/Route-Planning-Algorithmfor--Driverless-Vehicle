import copy
import random
import utils
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# INF无穷大
INF = 100000


class K_M:
    def __init__(self,express_site_num,car_num, start_site,express_sites,cars_driving_distance):
        self.car_num=car_num    #the number of DDV
        self.start_site=start_site  #The serial number of SC
        self.express_site_num=express_site_num  #the number of ES
        self.express_sites=express_sites    #Location of ESs
        self.cars_driving_distance=cars_driving_distance    #The distance vector that DDV can travel

    def cal_dis(self,data, clu, k): #Calculate the distance between the centroid and the data point
        dis = []
        for i in range(len(data)):
            dis.append([])
            for j in range(k):
                dis[i].append(math.sqrt((data[i, 0] - clu[j, 0]) ** 2 + (data[i, 1] - clu[j, 1]) ** 2))
        return np.asarray(dis)

    def group(self,data, dis): #grouping points
        clusterRes = [0] * len(data)
        for i in range(len(data)):
            seq = np.argsort(dis[i])
            clusterRes[i] = seq[0]

        return np.asarray(clusterRes)

    def cal_cen(self,data, clusterRes, k):#calculate centroid
        clunew = []
        for i in range(k):
            # 计算每个组的新质心
            idx = np.where(clusterRes == i)
            sum = data[idx].sum(axis=0)
            avg_sum = sum / len(data[idx])
            clunew.append(avg_sum)
        clunew = np.asarray(clunew)
        return clunew[:, 0: 2]

    def classfy(self,data, clu, k):#Iterative convergence update centroid
        clulist = self.cal_dis(data, clu, k)
        clusterRes = self.group(data, clulist)
        clunew = self.cal_cen(data, clusterRes, k)
        diff = clunew - clu
        return diff, clunew, k, clusterRes

    def get_closest_dist(self,point,cents):
        min_dist=INF
        for i,cent in enumerate(cents):
            dist = math.sqrt((cent[0] - point[0]) ** 2 + (cent[1] - point[1]) ** 2)
            if dist < min_dist:
                min_dist=dist
        return min_dist


    def get_k_clu(self):
        k = self.car_num  # 类别个数
        data_copy = np.array(self.express_sites)
        data = data_copy[0:self.express_site_num - 1, 1:3]  # 删除最后一行并且删除第一列
        d=[0 for i in range(len(data))]
        clu=[]
        clu.append(random.choice(data[:, 0:2].tolist()))
        for i in range(1,k):
            total=0.0
            for i,point in enumerate(data[:, 0:2].tolist()):
                d[i]=self.get_closest_dist(point,clu) #计算各个样本点到最近聚类中心的距离，为后期分配选中概率
                total+=d[i]
            total*=random.random()
            for i,di in enumerate(d):
                total-= di
                if total>0:
                    continue
                clu.append(data[i])
                break
        return clu

    def k_means_starts(self):
        #print("进入K-Means++算法阶段")
        k = self.car_num  # 类别个数
        data_copy =np.array(self.express_sites)
        data=data_copy[0:self.express_site_num-1,1:3] #删除最后一行并且删除第一列

        clu = self.get_k_clu()  # k-means++取质心
        clu = np.asarray(clu)
        diff, clunew, k, clusterRes = self.classfy(data, clu, k)
        while np.any(abs(diff) > 0):
            # print(clunew)
            diff, clunew, k, clusterRes = self.classfy(data, clunew, k)

        clulist = self.cal_dis(data, clunew, k)  # 质心与样本点的距离矩阵
        clusterResult = self.group(data, clulist)

        new_data = []
        for i in range(len(data)):
            da = np.append(data[i], [i + 1, clusterResult[i]]) #往数组中添加站点编号和分类编号
            new_data.append(da)
        new_data = np.array(new_data)
        new_data = new_data[new_data[:, 3].argsort()] #按类别号进行排序
        category = list(new_data[:, 3]) #取出第4列的分类向量
        count = []  #存放各分类的站点数
        for i in range(self.car_num):
            count.append(category.count(i))
        cumulate_count = copy.deepcopy(count) #cumulate_count为数量累积向量
        for i in range(self.car_num):
            if (i != 0):
                cumulate_count[i] = cumulate_count[i] + cumulate_count[i - 1]


        chrom = list(new_data[:, 2])  #取站点编号为染色体排列

        kk = [] #将整个染色体分为5个快递车路线，便于之后进行排序
        for i in range(self.car_num):
            if (i == 0):
                mm = chrom[0:cumulate_count[i]]
            else:
                mm = chrom[cumulate_count[i - 1]:cumulate_count[i]]
            kk.append(mm)


        sort_dis=sorted(self.cars_driving_distance) #对行驶距离进行排序

        pp=[]   #得到行驶距离的排序下标  例如[1000,2600,2200,3000,1700] 得到的pp为[0,3,2,4,1]
        for i in range(5):
            for j in range(5):
                if(self.cars_driving_distance[i]==sort_dis[j]):
                    pp.append(j)



        seq=np.argsort(count)

        new_chrom=[]
        new_count=[]
        for i in pp:
            new_count.append(len(kk[seq[i]]))
            new_chrom.extend(kk[seq[i]])

        new_cumulate_count = copy.deepcopy(new_count)  # new_cumulate_count为数量累积向量
        for i in range(self.car_num):
            if (i != 0):
                new_cumulate_count[i] = new_cumulate_count[i] + new_cumulate_count[i - 1]



        f=[x for x in range(self.express_site_num + 1,self.express_site_num + self.car_num)] #间断点
        for i in range(self.car_num-1):
            chrom.insert(new_cumulate_count[i], f[i]) #在类别间断点处添加间断点
        chrom_new = list(map(int, chrom)) #将染色体转化为列表后返回
        return chrom_new #返回的是经过排序后的染色体




class GA:
    def __init__(self, car_num, express_site_num, start_site,max_iteration_num1,
                 max_iteration_num2,distance_weight,diff_weight,time_weight):
        self.population_size = 30
        self.population = []  # population
        self.car_num = car_num  # the number of DDV
        self.express_site_num = express_site_num  # the number of ES
        self.chrom_len = car_num + express_site_num - 2  # Chromosome length=(ES-1) + (DDV-1)
        self.start_site = start_site  # SC
        self.max_iteration_num1 = max_iteration_num1  # stage 1
        self.max_iteration_num2 = max_iteration_num2 # stage 2
        self.iteration_count = 1
        self.break_points = [x for x in range(self.express_site_num + 1,self.express_site_num + self.car_num)]  # discontinuities
        self.express_sites = []  # location [ [1, 116.407526, 39.904033],[],...,[]]
        self.disMatrix = np.zeros([express_site_num,express_site_num])  # (distance Matrix)
        self.conMatrix=np.zeros([express_site_num,express_site_num])  # (congestion Matrix)
        self.per_gen_best_chrom = []  # The optimal individual for each generation
        self.per_gen_best_chrom_fit = 0  # Fitness of the optimal individual per generation
        self.per_gen_best_path = []  # The path of the optimal individual in each generation
        self.per_gen_best_dis_sum = INF  # The total distance of the optimal individuals in each generation
        self.per_gen_passing_sites=[]  #The number of express stations of all express vehicle routes in each generation of optimal individuals
        self.per_gen_best_dist_list=[] #List of optimal route length for each generation (record the distance traveled for each route)
        self.all_per_gen_best_chrom = []  # Record the optimal individual change of each generation in each iteration process
        self.all_per_gen_best_chrom_fit = []  # The fitness changes of the optimal individuals in each generation were recorded during each iteration
        self.all_per_pop_best_dist_sum = []  # Record the total distance variation of the optimal individual in each generation during each iteration
        self.best_chrom = []  # The global optimal individual, not necessarily the optimal individual in each generation, the optimal individual in each generation may be worse than the optimal individual in the past
        self.best_chrom_fit = 0  # Fitness of globally optimal individuals
        self.best_path = []  # Path of globally optimal individual
        self.best_dis_sum = INF  # Sum of paths of globally optimal individuals
        self.best_passing_sites=[] #The number of express stations of all express vehicle paths in the global optimal individual
        self.best_dist_list=[] #Global Optimal solution Route Length list (record the distance traveled for each route)
        self.all_best_chrom = []  # Record the change of the global optimal individual during each iteration
        self.all_best_chrom_fit = []  # The fitness changes of the global optimal individual during each iteration were recorded
        self.all_best_dist_sum = []  # Record the total distance of the globally optimal individual during each iteration

        self.cross_rate = 0.8  # cross
        self.mutation_rate = 0.25  # mutatuin

        self.cross_ox_rate = 0.5  # OX_rate

        self.mutation_inverse_rate = 0.3
        self.mutation_reverse_rate = 0.2
        self.mutation_insert_rate = 0.3
        self.mutation_swap_rate = 0.2

        self.cars_driving_distance = []
        self.driving_time=[]

        self.distance_weight = distance_weight
        self.diff_weight=diff_weight
        self.time_weight=time_weight

        self.best_time_list=[]
        self.old_route_list=[]

        self.new_time = 0.0
        self.old_time = 0.0

        self.start_index=0

    def get_sites(self):
        express_sites = []
        sites_data_1000 = pd.read_csv('./data/express_site5.csv').values
        m = sites_data_1000[:, 0:1]
        m = [j for i in m for j in i]
        x = sites_data_1000[:, 1:2]
        x = [j for i in x for j in i]
        y = sites_data_1000[:, 2:3]
        y = [j for i in y for j in i]
        for i in range(len(x)):
            express_sites.append([m[i], x[i], y[i]])
        print(express_sites)
        return express_sites

    def get_driving_distance(self):
        cars_driving_distance = []
        cars_data_5 = pd.read_csv('./data/driving_distance_new.csv').values
        x = cars_data_5[:, 1:2]
        x = [j for i in x for j in i]
        cars_driving_distance = x
        print(cars_driving_distance)
        return cars_driving_distance

    def D(slef, location1, location2):  # 计算快递站点间的距离步骤1
        return math.sqrt(pow(location1[1] - location2[1], 2) + pow(location1[2] - location2[2], 2))

    def DMAT(self, locations):  # 计算快递站点间的距离步骤2  locations是所有快递站点的坐标信息，返回的是距离矩阵
        length = len(locations)
        distance = np.ones([length, length])
        # print(distance.shape)
        for i in range(length):
            for j in range(length):
                distance[i, j] = self.D(locations[i], locations[j])
        #print(distance)
        return distance

    def get_congestion_situation(self):
        my_matrix = np.loadtxt(open('./data/congestion_situation_1000.csv'),delimiter=",",skiprows=1)
        #print(my_matrix)
        return my_matrix

    def express_sites_init_pop(self):
        # 读取中国快递站点文件，并初始化express_sites
        self.express_sites = self.get_sites()
        # 获得各快递站点间的距离矩阵
        self.disMatrix = self.DMAT(self.express_sites)
        # print(self.DMAT(self.express_sites))
        # 读取各快递车行能行驶的距离
        self.cars_driving_distance = self.get_driving_distance()
        #print(self.cars_driving_distance)
        #读取各快递点间道路的拥堵情况 数值范围：1~2   1表示无拥堵，2表示拥堵情况很严重  （1-2）之间根据值大小表示拥堵情况的大小
        self.conMatrix=self.get_congestion_situation()

        k_means=K_M(self.express_site_num,self.car_num,self.start_site,self.express_sites,self.cars_driving_distance)

        print("Enter the k-means ++ algorithm stage")
        for i in range(self.population_size):  # population_size为种群的大小：即有多少条路径
            # 注意：快递站点起点从1开始，而不是从0
            chrom = k_means.k_means_starts()
            self.population.append(chrom)
        # 初始化全局最优个体和它的适应度
        self.best_chrom = self.population[0]
        self.best_chrom_fit = self.fitness_function(self.best_chrom,1)

    def select_better_pop(self, population,choice):
        new_population = []  # 下一代种群
        for i in range(self.population_size):
            # 随机选择2个个体
            competitors = random.choices(population, k=2)
            # 选择适应度大的个体
            winner = max(competitors, key=lambda x: self.fitness_function(x,choice))
            new_population.append(winner)
        return new_population

    def cross_ox(self, parent_chrom1, parent_chrom2,chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index, chrom_length - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        # temp_gene1为parent_chrom1被选中的染色体片段[index1:index2)
        temp_gene1 = parent_chrom1[index1:index2]
        # temp_gene2为parent_chrom2被选中的染色体片段[index1:index2)
        temp_gene2 = parent_chrom2[index1:index2]

        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0
        # 得到child_chrom1
        for i in parent_chrom2:
            if child_p1 == index1:
                child_chrom1.extend(temp_gene1)
                child_p1 += 1
            if i not in temp_gene1:
                child_chrom1.append(i)
                child_p1 += 1

        # 得到child_chrom2
        for i in parent_chrom1:
            if child_p2 == index1:
                child_chrom2.extend(temp_gene2)
                child_p2 += 1
            if i not in temp_gene2:
                child_chrom2.append(i)
                child_p2 += 1
        return child_chrom1, child_chrom2

    def cross_pmx(self, parent_chrom1, parent_chrom2,chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index, chrom_length - 1)
        if index1 > index2:
            index1, index2 = index2, index1

        parent_part1, parent_part2 = parent_chrom1[index1:index2], parent_chrom2[index1:index2]
        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0  # 指针用来解决复制到指定位置问题
        # 子代1
        for i in parent_chrom1:
            # 指针到达父代的选中部分
            if index1 <= child_p1 < index2:
                # 将父代2选中基因片段复制到子代1指定位置上
                child_chrom1.append(parent_part2[child_p1 - index1])
                child_p1 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p1 < index1 or child_p1 >= index2:
                # 父代1未选中部分含有父代2选中部分基因
                if i in parent_part2:
                    tmp = parent_part1[parent_part2.index(i)]
                    while tmp in parent_part2:
                        tmp = parent_part1[parent_part2.index(tmp)]
                    child_chrom1.append(tmp)
                elif i not in parent_part2:
                    child_chrom1.append(i)
                child_p1 += 1
        # 子代2
        for i in parent_chrom2:
            # 指针到达父代的选中部分
            if index1 <= child_p2 < index2:
                # 将父代1选中基因片段复制到子代2指定位置上
                child_chrom2.append(parent_part1[child_p2 - index1])
                child_p2 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p2 < index1 or child_p2 >= index2:
                # 父代2未选中部分含有父代1选中部分基因
                if i in parent_part1:
                    tmp = parent_part2[parent_part1.index(i)]
                    # 解决1<->6<->3
                    while tmp in parent_part1:
                        tmp = parent_part2[parent_part1.index(tmp)]
                    child_chrom2.append(tmp)
                elif i not in parent_part1:
                    child_chrom2.append(i)
                child_p2 += 1
        return child_chrom1, child_chrom2

    def crossover(self, population,chrom_length,choice):
        # 交叉:比较特殊，只有PMX和OX
        new_population = []
        # “杰出选择”选择出新的一代
        selected_pop = self.select_better_pop(population,choice)  # 传入的是种群，返回的是数量相同的种群，但是里面的个体相对优化了
        for i in range(int(self.population_size / 2)):  # -------------------------改：除以2 不然返回的种群数量就变成60了
            rate = random.random()  # 随机数，决定是PMX还是OX
            two_chrom = random.choices(selected_pop, k=2)  # 随机选取两次
            if rate <= self.cross_ox_rate:  # 小于交叉选择顺序匹配交叉OX的概率
                # 执行OX
                child_chrom1, child_chrom2 = self.cross_ox(two_chrom[0], two_chrom[1],chrom_length)
                new_population.append(child_chrom1)
                new_population.append(child_chrom2)
            else:
                # 执行PMX
                child_chrom1, child_chrom2 = self.cross_pmx(two_chrom[0], two_chrom[1],chrom_length)
                new_population.append(child_chrom1)
                new_population.append(child_chrom2)
        return new_population

    def mutate_swap(self, parent_chrom,chrom_length):
        # 如果index1和index2相等，则交换变异相当于没有执行
        index1 = random.randint(self.start_index, chrom_length - 1)
        index2 = random.randint(self.start_index, chrom_length - 1)
        child_chrom = parent_chrom[:]
        child_chrom[index1], child_chrom[index2] = child_chrom[index2], child_chrom[index1]
        return child_chrom

    def mutate_reverse(self, parent_chrom, chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index, chrom_length - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        child_chrom = parent_chrom[:]
        tmp = child_chrom[index1: index2]
        tmp.reverse()
        child_chrom[index1: index2] = tmp
        return child_chrom

    def mutate_inverse(self, parent_chrom, chrom_length):
        tmp_chrom = parent_chrom[:]
        # 将增加的切断点还原成起始点
        for i in range(len(parent_chrom)):
            if parent_chrom[i] in self.break_points:
                tmp_chrom[i] = self.start_site
        index1 = random.randint(self.start_index , chrom_length-1)
        child_chrom = parent_chrom[:]
        mindistance = INF
        minindex=index1+1
        index2=index1+1

        while(index2!=chrom_length and tmp_chrom[index2]!=self.start_site):
            if (self.disMatrix[tmp_chrom[index1] - 1][tmp_chrom[index2] - 1] < mindistance):
                minindex = index2
                mindistance = self.disMatrix[tmp_chrom[index1] - 1][tmp_chrom[index2] - 1]
            index2+=1
        temp = child_chrom[index1 + 1:minindex]
        temp.reverse()
        child_chrom[index1 + 1:minindex] = temp
        return child_chrom




    def mutate_insert(self, parent_chrom,chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index, chrom_length - 1)
        child_chrom = parent_chrom[:]
        child_chrom.pop(index2)
        child_chrom.insert(index1 + 1, parent_chrom[index2])
        return child_chrom

    def mutate_insert2(self, parent_chrom,chrom_length):
        tmp_chrom = parent_chrom[:]
        # 将增加的切断点还原成起始点
        for i in range(len(parent_chrom)):
            if parent_chrom[i] in self.break_points:
                tmp_chrom[i] = self.start_site

        index1 = random.randint(self.start_index, chrom_length - 1)
        child_chrom = parent_chrom[:]
        mindistance = INF
        index2 = random.randint(index1, chrom_length - 1)
        minindex = index2

        while (index2 != chrom_length and tmp_chrom[index2] != self.start_site):
            if (self.disMatrix[tmp_chrom[index1] - 1][tmp_chrom[index2] - 1] < mindistance):
                minindex = index2
                mindistance = self.disMatrix[tmp_chrom[index1] - 1][tmp_chrom[index2] - 1]
            index2 += 1

        child_chrom.pop(minindex)
        child_chrom.insert(index1 + 1, parent_chrom[minindex])
        return child_chrom

    def mutation(self, population,chrom_length):
        rate_sum = []
        rate_sum.extend([self.mutation_inverse_rate,
                         self.mutation_inverse_rate+self.mutation_insert_rate,
                         self.mutation_inverse_rate+self.mutation_insert_rate+self.mutation_reverse_rate, 1])
        new_population = []
        for i in range(self.population_size):
            p = random.random()
            if p <= rate_sum[0]:  # 0.4的概率
                # 启发式逆转
                child_chrom = self.mutate_inverse(population[i],chrom_length)
                new_population.append(child_chrom)
            elif p <= rate_sum[1]:  # 0.3的概率
                # 逆序变异
                child_chrom = self.mutate_insert2(population[i],chrom_length)
                new_population.append(child_chrom)
            elif p <= rate_sum[2]:  # 0.2的概率
                # 插入变异
                child_chrom = self.mutate_reverse(population[i],chrom_length)
                new_population.append(child_chrom)
            else:  # 0.1的概率
                # 交换变异
                child_chrom = self.mutate_swap(population[i],chrom_length)
                new_population.append(child_chrom)
        return new_population

    def compute_pop_fitness(self, population,choice):
        return [self.fitness_function(chrom,choice) for chrom in population]  # 调用适应度函数

    def get_best_chrom(self, population,choice):
        tmp = self.compute_pop_fitness(population,choice)
        index = tmp.index(max(tmp))
        return population[index], index

    def get_cars_distance(self, chrom):  # 获得各个快递车的行驶距离
        tmp_chrom = chrom[:]
        # 将增加的切断点还原成起始点
        for i in range(len(chrom)):
            if chrom[i] in self.break_points:
                tmp_chrom[i] = self.start_site
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个快递车路线，可以为空
        all_routines = []  # 所有快递车路线
        passing_sites_count = []
        for v in tmp_chrom:
            if v == self.start_site:
                all_routines.append(one_routine)
                passing_sites_count.append(len(one_routine))
                one_routine = []
            elif v != self.start_site:
                one_routine.append(v)
        # 还有一次需要添加路线
        all_routines.append(one_routine)
        passing_sites_count.append(len(one_routine))
        routines_dis = []  # 所有路径总距离组成的列表
        # 计算每一条路总的距离
        for r in all_routines:
            # print(r)
            # print(r[0])
            distance = 0
            # 有一个快递车路线为空列表，即一个快递车不出门
            if len(r) == 0:
                distance = 0.0
                routines_dis.append(distance)
            else:
                r_len = len(r)
                for i in range(r_len):
                    # 别忘了最后加上起始点到第一个点的距离
                    if i == 0:
                        distance += self.disMatrix[self.start_site - 1, r[i] - 1]
                    if i + 1 < r_len:
                        #print("%d %d %f" %(r[i] ,r[i+1], self.disMatrix[r[i] - 1, r[i + 1] - 1] ))
                        distance += self.disMatrix[r[i] - 1, r[i + 1] - 1]
                    # 最后一个顶点，下一站是起始点
                    elif i == r_len - 1:
                        distance += self.disMatrix[r[i] - 1, self.start_site - 1]
                routines_dis.append(distance)
        #print(routines_dis)
        return all_routines, routines_dis, passing_sites_count

    def get_cars_distance_and_time(self, chrom):  #新增方法
        length=len(chrom) #这是途径快递站点的数目
        distance = 0.0
        time=0.0
        if(length!=0):
            for i in range(length):
                # 别忘了最后加上起始点到第一个点的距离
                if i == 0:
                    distance += self.disMatrix[self.start_site - 1, chrom[i] - 1]
                    time += (self.disMatrix[self.start_site - 1, chrom[i] - 1] / 30) * self.conMatrix[
                        self.start_site - 1, chrom[i] - 1]
                if i + 1 < length:
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                    time += (self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1] / 30) * self.conMatrix[
                        chrom[i] - 1, chrom[i + 1] - 1]
                # 最后一个顶点，下一站是起始点
                elif i == length - 1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_site - 1]
                    time += (self.disMatrix[chrom[i] - 1, self.start_site - 1] / 30) * self.conMatrix[
                        chrom[i] - 1, self.start_site - 1]
        return distance,time

    def get_cars_distance_and_time2(self, chrom):  #新增方法
        length=len(chrom) #这是途径快递站点的数目
        distance = 0.0
        time=0.0
        if(length!=0):
            for i in range(length):
                # 别忘了最后加上起始点到第一个点的距离
                if i == 0:
                    distance += self.disMatrix[self.start_site - 1, chrom[i] - 1]
                if i + 1 < length:
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                # 最后一个顶点，下一站是起始点
                elif i == length - 1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_site - 1]
            time+=self.new_time
            for i in range(self.start_index,length):

                if i+1 <= length:
                    time += (self.disMatrix[chrom[i-1] - 1, chrom[i] - 1] / 30) * self.conMatrix[
                        chrom[i-1] - 1, chrom[i] - 1]
                elif i== length :
                    time += (self.disMatrix[chrom[i-1] - 1, self.start_site - 1] / 30) * self.conMatrix[
                        chrom[i] - 1, self.start_site - 1]
        return distance,time

    def obj_function(self, chrom,choice): #返回的值obj越小表示个体越好
        if(choice==1):
            a, routines_dis, c = self.get_cars_distance(chrom)
            routines_dis=np.array(routines_dis)
            obj1 = 0
            if((routines_dis==0).any()):
                obj1+=5000
            sum_path = sum(routines_dis)
            obj1 += self.distance_weight * sum_path
            for i in range(len(self.cars_driving_distance)):
                if (routines_dis[i] > self.cars_driving_distance[i]):
                    diff = (routines_dis[i] - self.cars_driving_distance[i]) / routines_dis[i]
                    obj1 += diff * self.diff_weight
            return obj1
        elif(choice==2):
            distance=0.0
            time=0.0
            for i in range(len(chrom)):
                # 别忘了最后加上起始点到第一个点的距离
                if i == 0:
                    distance += self.disMatrix[self.start_site - 1, chrom[i] - 1]
                    time+=(self.disMatrix[self.start_site - 1, chrom[i] - 1]/30) * self.conMatrix[self.start_site - 1, chrom[i] - 1]
                if i + 1 < len(chrom):
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                    time+=(self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]/30)*self.conMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                # 最后一个顶点，下一站是起始点
                elif i == len(chrom) - 1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_site - 1]
                    time+=(self.disMatrix[chrom[i] - 1, self.start_site - 1]/30) * self.conMatrix[chrom[i] - 1, self.start_site - 1]
            obj2=self.distance_weight * distance + time * self.time_weight
            return obj2
        elif(choice==3):
            distance = 0.0
            time =0.0
            time +=self.new_time
            for i in range(len(chrom)):
                if i==0:
                    distance += self.disMatrix[self.start_site - 1, chrom[i] - 1]
                if i+1 <len(chrom):
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                elif i== len(chrom) -1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_site - 1]
            for i in range(self.start_index,len(chrom)):
                if i + 1 < len(chrom):
                    time+=(self.disMatrix[chrom[i-1]-1,chrom[i]-1]/30)* self.conMatrix[
                        chrom[i-1] - 1, chrom[i] - 1]
                elif i == len(chrom) :
                    time += (self.disMatrix[chrom[i - 1] - 1, self.start_site - 1] / 30) * self.conMatrix[
                        chrom[i - 1] - 1, self.start_site - 1]
            obj3 = self.distance_weight * distance + time * self.time_weight
            return obj3


    def fitness_function(self, chrom,choice):
        return math.exp(1.0 / self.obj_function(chrom,choice))

    def ga_process(self):
        """
        针对实验 2：运送快递
        GA流程
        """
        self.express_sites_init_pop()  # 初始化种群
        self.ga_process_iterator(self.get_cars_distance)  # 调用GA算法的迭代过程


        print("******************************")
        print(self.max_iteration_num1 + 1)
        print(self.all_best_dist_sum)
        print("******************************")

        self.start_route_planning() #根据道路信息进行路径重新规划

        self.route_planning() #根据实时道路信息进行路径重新规划

        # self.print_best_routine() #打印路径

    def ga_process_iterator(self, get_distance_func):
        # 遗传算法的迭代过程1
        print("Enter the first phase of genetic algorithm")
        pop_new=[]
        while self.iteration_count < self.max_iteration_num1:
            # 杰出选择 先调出好的个体重新组成一个种群
            pop_new = self.select_better_pop(self.population,1)
            # -------------------变异------------------------------------------
            # 随机数决定是否变异
            p_mutate = random.random()
            if p_mutate <= self.mutation_rate:
                pop_new = self.mutation(pop_new,self.chrom_len)
            # -------------------新的一代有关参数更新-------------------------------
            # *******************新的一代的最优个体有关参数更新**********************
            # 计算种群所有个体的适应度
            pop_fitness_list = self.compute_pop_fitness(pop_new,1)
            # 每代最优个体per_gen_best_chrom及其在种群中的下标best_index
            self.per_gen_best_chrom, best_index = self.get_best_chrom(pop_new,1)
            # 每代最优个体的适应度
            self.per_gen_best_chrom_fit = pop_fitness_list[best_index]
            # 每代最优个体最好的路径组成和每条路路径长度per_gen_best_dist_list以及每条路径途经的站点数self.per_gen_passing_sites
            self.per_gen_best_path, self.per_gen_best_dist_list, self.per_gen_passing_sites = get_distance_func(
                self.per_gen_best_chrom)
            # 每代最优个体所有快递车路线之和
            self.per_gen_best_dis_sum = sum(self.per_gen_best_dist_list)

            # 记录下每代最优个体
            self.all_per_gen_best_chrom.append(self.per_gen_best_chrom)
            # 记录下每代最优个体的适应度
            self.all_per_gen_best_chrom_fit.append(self.per_gen_best_chrom_fit)
            # 记录每次迭代过程中每代最优个体的总距离变化情况
            self.all_per_pop_best_dist_sum.append(self.per_gen_best_dis_sum)
            # *******************全局最优个体有关参数更新****************************
            # 每代最优个体与全局最优个体根据适应度比较，如果每代最优个体适应度更小，则更新全局最优个体
            if self.per_gen_best_chrom_fit > self.best_chrom_fit:
                self.best_chrom = self.per_gen_best_chrom
                self.best_chrom_fit = self.per_gen_best_chrom_fit
                # 全局最优个体最好的路径组成和每条路路径长度
                self.best_path, self.best_dist_list, self.best_passing_sites = get_distance_func(self.best_chrom)
                # self.best_path = self.per_gen_best_path
                # 全局最优个体的路径之和
                self.best_dis_sum = self.per_gen_best_dis_sum
                # 记录下每次迭代过程中全局最优个体
                self.all_best_chrom.append(self.best_chrom)

            # 记录每次迭代过程中全局最优个体的适应度变化情况
            self.all_best_chrom_fit.append(self.best_chrom_fit)
            # 记录每次迭代过程中全局最优个体的总距离
            self.all_best_dist_sum.append(self.best_dis_sum)

            # 输出
            if self.iteration_count % 50 == 0:
                self.print_iteration()

            # *******************种群有关参数更新****************************
            #加入判断，判断是快递车是否能开这就远的距离
            flag = True
            if (len(self.best_dist_list) != 0):
                for i in range(len(self.best_dist_list)):
                    if (self.best_dist_list[i] > self.cars_driving_distance[i]):
                        flag = False
            else:
                flag = False

            if (flag == False):
                # 计数器加1
                self.iteration_count += 1
            elif (flag == True and self.iteration_count != self.max_iteration_num1):
                self.max_iteration_num1 = self.iteration_count
                self.print_iteration()
            elif (self.iteration_count == self.max_iteration_num1 - 1):
                self.iteration_count += 1
                self.print_iteration()
            # 更新种群
            self.population = pop_new
            # -------------------新的一代有关参数更新结束------------------------------------------------

    def start_route_planning(self):
        print("Enter the second stage of genetic algorithm")
        route = []  # 记录最终所有快递车的路径
        route_list = []  # 记录最终所有快递车行驶的距离
        for i in range(len(self.best_path)):
            population=[] #存放单个快递车走的路径
            population.append(self.best_path[i])
            for j in range(self.population_size-1):  # population_size为种群的大小：即有多少条路径
                chrom=[x for x in self.best_path[i]]
                random.shuffle(chrom)
                population.append(chrom)
            best_chrom = population[0]
            best_chrom_fit = self.fitness_function(best_chrom,2)
            pop_new=[]
            iteration_count=1
            best_distance, best_time = self.get_cars_distance_and_time(best_chrom)
            while iteration_count <= self.max_iteration_num2+int(len(best_chrom)/100)*100:
                # 杰出选择 先调出好的个体重新组成一个种群
                pop_new = self.select_better_pop(population, 2)
                # -------------------交叉------------------------------------------
                # 随机数决定是否交叉
                p_cross = random.random()
                if p_cross <= self.cross_rate:
                    pop_new = self.crossover(pop_new,len(best_chrom),2)
                # -------------------变异------------------------------------------
                # 随机数决定是否变异
                p_mutate = random.random()
                if p_mutate <= 0.25:
                    pop_new = self.mutation(pop_new,len(best_chrom))
                # -------------------新的一代有关参数更新-------------------------------
                # 计算种群所有个体的适应度
                pop_fitness_list = self.compute_pop_fitness(pop_new, 2)
                # 每代最优个体per_gen_best_chrom及其在种群中的下标best_index
                per_gen_best_chrom, best_index = self.get_best_chrom(pop_new, 2)
                # 每代最优个体的适应度
                per_gen_best_chrom_fit = pop_fitness_list[best_index]
                #每代最优个体(单量快递车)行驶的距离和事件
                per_gen_best_distance,per_gen_best_time=self.get_cars_distance_and_time(per_gen_best_chrom)
                # *******************全局最优个体有关参数更新****************************
                # 每代最优个体与全局最优个体根据适应度比较，如果每代最优个体适应度更小，则更新全局最优个体

                if per_gen_best_chrom_fit > best_chrom_fit:
                    best_chrom = per_gen_best_chrom
                    best_chrom_fit = per_gen_best_chrom_fit
                    best_distance=per_gen_best_distance
                    best_time=per_gen_best_time
                # 输出
                if iteration_count % 50 == 0:
                    print("route %d is iterated %d times" % (i + 1, iteration_count))
                    print("The driving distance of DDV:%f，The total travel time of DDV：%f" % (best_distance, best_time))
                    print("The global optimal solution route is{}".format(best_chrom))
                    print("---------------------------------------------------------")
                    print("The driving distance of DDV：%f，The total travel time of DDV：%f" % (
                    per_gen_best_distance, per_gen_best_time))
                    print("The optimal solution route of each generation is{}".format(per_gen_best_chrom))
                    print("**************************************************************************")
                # *******************种群有关参数更新****************************
                # 更新种群
                population = pop_new
                # 计数器加1
                iteration_count += 1
                # -------------------新的一代有关参数更新结束------------------------------------------------
            route.append(best_chrom)
            route_list.append(best_distance)
            self.best_time_list.append(best_time)
        #这里再加上打印路径即可
        self.best_dis_sum=sum(route_list)
        self.best_dist_list=route_list
        self.best_path=route
        self.print_best_routine()
        self.old_route_list=route_list



    def route_planning(self):
        print("Enter the third stage: real-time route planning stage")
        route = []  # 记录最终所有快递车的路径
        route_list = []  # 记录最终所有快递车行驶的距离
        new_path = list(self.best_path)
        old_time_list = [] #记录第二阶段的最好个体的行驶时间
        new_time_list = []
        all_old_chrom=self.best_path
        for i in range(len(self.best_path)):  # 5个快递车
            chrom=[]
            chrom = self.best_path[i]
            old_chrom=all_old_chrom[i]
            best_chrom=[]
            best_distance=[]
            self.new_time=0.0
            self.old_time=0.0
            for k in range(len(chrom)+1):  # 第i个快递车途径的站点数
                self.start_index=k
                if(k==0):
                    print("The %d ES on the %d DDV route is %d" %(k+1, i+1, chrom[k]))
                    self.new_time += (self.disMatrix[self.start_site - 1, chrom[k] - 1] / 30) * self.conMatrix[
                        self.start_site - 1, chrom[k] - 1]
                    self.old_time +=(self.disMatrix[self.start_site - 1, old_chrom[k] - 1] / 30) * self.conMatrix[
                        self.start_site - 1, old_chrom[k] - 1]
                    continue
                self.conMatrix=np.random.uniform(1,5,size=(1001,1001))
                if (k + 1 < len(chrom) and self.conMatrix[chrom[k-1] - 1, chrom[k] - 1] <= 3):
                    print("The %d ES on the %d DDV route is %d" % (k + 1, i + 1, chrom[k]))
                    self.new_time+=(self.disMatrix[chrom[k-1]-1,chrom[k]-1]/30)*self.conMatrix[chrom[k-1]-1,chrom[k]-1]
                    self.old_time+=(self.disMatrix[old_chrom[k-1]-1,old_chrom[k]-1]/30)*self.conMatrix[old_chrom[k-1]-1,old_chrom[k]-1]
                    continue
                if (k+1==len(chrom)):
                    print("The %d ES on the %d DDV route is %d" % (k + 1, i + 1, chrom[k]))
                    self.new_time += (self.disMatrix[chrom[k-1] - 1, chrom[k]-1] / 30) * self.conMatrix[
                        chrom[k-1] - 1, chrom[k] - 1]
                    self.old_time +=(self.disMatrix[old_chrom[k-1] - 1, old_chrom[k]-1] / 30) * self.conMatrix[
                        old_chrom[k-1] - 1, old_chrom[k] - 1]
                    continue
                if(k==len(chrom)):
                    print("The %d DDV completes delivery and heads to the sorting center：" % (i + 1))
                    self.new_time += (self.disMatrix[chrom[k-1] - 1, self.start_site-1] / 30) * self.conMatrix[
                        chrom[k-1] - 1, self.start_site - 1]
                    self.old_time +=(self.disMatrix[old_chrom[k-1] - 1, self.start_site-1] / 30) * self.conMatrix[
                        old_chrom[k-1] - 1, self.start_site - 1]
                    continue
                print("The road ahead is congested and the route planning is being replanned")

                population = []  # 存放单个快递车走的路径
                for j in range(self.population_size):  # population_size为种群的大小：即有多少条路径
                    chro = [x for x in chrom]
                    population.append(chro)
                best_chrom = population[0]
                best_chrom_fit = self.fitness_function(best_chrom, 3)
                pop_new = []
                iteration_count = 1
                # best_distance = 0.0
                # best_time = 0.0
                best_distance, best_time = self.get_cars_distance_and_time2(best_chrom)
                while iteration_count <= int((len(chrom)-k)*2):
                    # 杰出选择 先调出好的个体重新组成一个种群
                    pop_new = self.select_better_pop(population, 3)
                    # -------------------交叉------------------------------------------
                    # 随机数决定是否交叉
                    p_cross = random.random()
                    if p_cross <= self.cross_rate:
                        pop_new = self.crossover(pop_new, len(best_chrom), 3)
                    # -------------------变异------------------------------------------
                    # 随机数决定是否变异
                    p_mutate = random.random()
                    if p_mutate <= 0.25:
                        pop_new = self.mutation(pop_new, len(best_chrom))
                    # -------------------新的一代有关参数更新-------------------------------
                    # 计算种群所有个体的适应度
                    pop_fitness_list = self.compute_pop_fitness(pop_new, 3)
                    # 每代最优个体per_gen_best_chrom及其在种群中的下标best_index
                    per_gen_best_chrom, best_index = self.get_best_chrom(pop_new, 3)
                    # 每代最优个体的适应度
                    per_gen_best_chrom_fit = pop_fitness_list[best_index]
                    # 每代最优个体(单量快递车)行驶的距离和事件
                    per_gen_best_distance, per_gen_best_time = self.get_cars_distance_and_time(per_gen_best_chrom)
                    # *******************全局最优个体有关参数更新****************************
                    # 每代最优个体与全局最优个体根据适应度比较，如果每代最优个体适应度更小，则更新全局最优个体

                    if per_gen_best_chrom_fit > best_chrom_fit:
                        best_chrom = per_gen_best_chrom
                        best_chrom_fit = per_gen_best_chrom_fit
                        best_distance = per_gen_best_distance
                        # best_time = per_gen_best_time
                    # 更新种群
                    population = pop_new
                    # 计数器加1
                    iteration_count += 1
                chrom=best_chrom
                # 输出
                print("Finish route planning for %d DDV" % (i + 1), end='')
                print("The %d ES on the %d DDV route is %d" % (k + 1, i + 1, chrom[k]))
                self.new_time += (self.disMatrix[chrom[k-1] - 1, chrom[k] - 1] / 30) * self.conMatrix[
                    chrom[k-1] - 1, chrom[k] - 1]
                self.old_time +=(self.disMatrix[old_chrom[k-1] - 1, old_chrom[k] - 1] / 30) * self.conMatrix[
                    old_chrom[k-1] - 1, old_chrom[k] - 1]
            route.append(best_chrom)
            route_list.append(best_distance)
            new_time_list.append(self.new_time)
            old_time_list.append(self.old_time)
            print("---------------------------------")
            print("The final route of %d DDV" % (i + 1), end='')
            print(best_chrom)
            print("new route%f old route%f" % (route_list[i], self.old_route_list[i]))
            print("---------------------------------")
        # 这里再加上打印路径即可
        self.best_dis_sum = sum(route_list)
        self.best_dist_list = route_list
        self.best_path = route
        self.print_best_routine()
        print("The driving time of the second stage{}".format(old_time_list), end='')
        print("The distance travelled in the second stage{}".format(self.old_route_list))
        print("The driving time of the third stage{}".format(new_time_list), end='')
        print("he distance travelled in the third stage{}".format(route_list))



    def print_iteration(self):
        print("After %d iterations" % self.iteration_count)
        print("The total distance traveled by all DDV:%f,The maximum distance traveled by DDV:%f" % (
        self.best_dis_sum, max(self.best_dist_list)))
        print("The global optimal solution route is{}".format(self.best_path))
        print("Number of passing sites{}".format(self.best_passing_sites))
        print("Global optimal solution route length list{}".format(self.best_dist_list))
        print("---------------------------------------------------------")
        print("The optimal solution distance of each generation：%f,The maximum distance traveled by DDV%f" % (
        self.per_gen_best_dis_sum, max(self.per_gen_best_dist_list)))
        print("The optimal solution route of each generation{}".format(self.per_gen_best_path))
        print("Number of passing sites{}".format(self.per_gen_passing_sites))
        print("Each generation optimal solution route length list{}".format(self.per_gen_best_dist_list))
        print("**************************************************************************")

    def print_best_routine(self):
        """
        打印快递车最优航线
        Returns:
        """
        print(type(self.best_path))
        print("DDVs' all route length：{}".format(self.best_dis_sum))
        #best_path, best_dist_list, passing_sites_count = self.get_cars_distance(self.best_chrom)
        # 打印全局最优个体的所有路线快递站点（包括起点和终点）
        for i in range(len(self.best_path)):
            print("The route length of {} DDV is {}".format(i + 1, self.best_dist_list[i]))
            print("The route of {} DDV：".format(i + 1), end="")
            if len(self.best_path[i]) == 0:
                print("The DDV does not leave")  # 这种情况可以通过设置目标函数避免
            else:
                for j in range(len(self.best_path[i])):
                    if j == 0:
                        print("{} ——> {} ".format(int(self.express_sites[self.start_site - 1][0]),
                                                  int(self.express_sites[self.best_path[i][j] - 1][0])), end="")
                    if j + 1 < len(self.best_path[i]):
                        print("——> {} ".format(int(self.express_sites[self.best_path[i][j + 1] - 1][0])), end="")
                    elif j == len(self.best_path[i]) - 1:
                        print("——> {}".format(int(self.express_sites[self.start_site - 1][0])))

if __name__ == "__main__":
    # 快递
    start = time.time()
    ga_obj = GA(car_num=5, express_site_num=1001, start_site=1001,max_iteration_num1=1500, max_iteration_num2=8000,
                distance_weight=1,diff_weight=1500,time_weight=250)
    ga_obj.ga_process()
    end = time.time()
    print("Time consuming:", end - start)