import copy
import random
import utils
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

INF = 100000

class K_M:
    def __init__(self,express_station_num,car_num, start_station,express_stations,cars_driving_distance):
        self.car_num=car_num    #the number of EDV
        self.start_station=start_station  #SC number
        self.express_station_num=express_station_num  #the number of ES
        self.express_stations=express_stations    #location [[1,x1,y1]...[1001,x1001,y1001]]，change to[[x1,y1]...[x1000,y1000]]
        self.cars_driving_distance=cars_driving_distance    #Distance vector

    def cal_dis(self,data, clu, k):
        dis = []
        for i in range(len(data)):
            dis.append([])
            for j in range(k):
                dis[i].append(math.sqrt((data[i, 0] - clu[j, 0]) ** 2 + (data[i, 1] - clu[j, 1]) ** 2))
        return np.asarray(dis)

    def group(self,data, dis):
        clusterRes = [0] * len(data)
        for i in range(len(data)):
            seq = np.argsort(dis[i])
            clusterRes[i] = seq[0]

        return np.asarray(clusterRes)

    def cal_cen(self,data, clusterRes, k):
        clunew = []
        for i in range(k):
            idx = np.where(clusterRes == i)
            sum = data[idx].sum(axis=0)
            avg_sum = sum / len(data[idx])
            clunew.append(avg_sum)
        clunew = np.asarray(clunew)
        return clunew[:, 0: 2]

    def classfy(self,data, clu, k):
        clulist = self.cal_dis(data, clu, k)
        clusterRes = self.group(data, clulist)
        clunew = self.cal_cen(data, clusterRes, k)
        diff = clunew - clu
        return diff, clunew, k, clusterRes

    def k_means_starts(self):
        k = self.car_num  # The number of categories
        data_copy =np.array(self.express_stations)
        data=data_copy[0:self.express_station_num-1,1:3]

        clu = random.sample(data[:, 0:2].tolist(), k)
        clu = np.asarray(clu)
        diff, clunew, k, clusterRes = self.classfy(data, clu, k)
        while np.any(abs(diff) > 0):
            diff, clunew, k, clusterRes = self.classfy(data, clunew, k)
        clulist = self.cal_dis(data, clunew, k)
        clusterResult = self.group(data, clulist)
        new_data = []
        for i in range(len(data)):
            da = np.append(data[i], [i + 1, clusterResult[i]]) #Add the site number and category number to the array
            new_data.append(da)
        new_data = np.array(new_data)
        new_data = new_data[new_data[:, 3].argsort()] #Sort by category number
        category = list(new_data[:, 3])
        count = []  # Number of sites for each category
        for i in range(self.car_num):
            count.append(category.count(i))
        cumulate_count = copy.deepcopy(count) #cumulate_count is Quantitative cumulative vector
        for i in range(self.car_num):
            if (i != 0):
                cumulate_count[i] = cumulate_count[i] + cumulate_count[i - 1]
        chrom = list(new_data[:, 2])  #Take the site number as chromosomal arrangement
        kk = [] #The entire chromosome was divided into five express truck routes for sorting later
        for i in range(self.car_num):
            if (i == 0):
                mm = chrom[0:cumulate_count[i]]
            else:
                mm = chrom[cumulate_count[i - 1]:cumulate_count[i]]
            kk.append(mm)
        sort_dis=sorted(self.cars_driving_distance) # Sort the distance traveled

        pp=[]   #Get the sort subscript of the driving distance  such as[1000,2600,2200,3000,1700] get pp is[0,3,2,4,1]
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

        new_cumulate_count = copy.deepcopy(new_count)
        for i in range(self.car_num):
            if (i != 0):
                new_cumulate_count[i] = new_cumulate_count[i] + new_cumulate_count[i - 1]

        f=[x for x in range(self.express_station_num + 1,self.express_station_num + self.car_num)]
        for i in range(self.car_num-1):
            chrom.insert(new_cumulate_count[i], f[i])
        chrom_new = list(map(int, chrom))
        return chrom_new

class GA:
    def __init__(self, car_num, express_station_num, start_station,max_iteration_num1,
                 max_iteration_num2,distance_weight,diff_weight,average_weight,time_weight):
        self.population_size = 30  
        self.population = []  # population
        self.car_num = car_num  # the number of ES
        self.express_station_num = express_station_num  # the number of ES
        self.chrom_len = car_num + express_station_num - 2  # Chromosome length=(ES-1) + (EDV-1)
        self.start_station = start_station   # SC
        self.max_iteration_num1 = max_iteration_num1   # stage 1
        self.max_iteration_num2 = max_iteration_num2 # stage 2
        self.iteration_count = 1  # Iteration counter
        self.break_points = [x for x in range(self.express_station_num + 1,self.express_station_num + self.car_num)]  # 切断点。切断点详细见下初始化函数check_vertex_init_population
        self.express_stations = []  # location [ [1, 116.407526, 39.904033],[],...,[]]
        self.disMatrix = np.zeros([express_station_num,express_station_num])  # (distance Matrix)
        self.conMatrix=np.zeros([express_station_num,express_station_num])  # (congestion Matrix)
        self.per_gen_best_chrom = []  # The optimal individual for each generation
        self.per_gen_best_chrom_fit = 0  # Fitness of the optimal individual per generation
        self.per_gen_best_path = []  # The path of the optimal individual in each generation
        self.per_gen_best_dis_sum = INF  #  The total distance of the optimal individuals in each generation
        self.per_gen_passing_stations=[]  #The number of express stations of all express vehicle routes in each generation of optimal individuals
        self.per_gen_best_dist_list=[] #List of optimal route length for each generation (record the distance traveled for each route)
        self.all_per_gen_best_chrom = []  # Record the optimal individual change of each generation in each iteration process
        self.all_per_gen_best_chrom_fit = [] # The fitness changes of the optimal individuals in each generation were recorded during each iteration
        self.all_per_pop_best_dist_sum = []  # Record the total distance variation of the optimal individual in each generation during each iteration
        self.best_chrom_fit = 0  # Fitness of globally optimal individuals
        self.best_path = []  # Path of globally optimal individual
        self.best_dis_sum = INF  # Sum of paths of globally optimal individuals
        self.best_passing_stations = []  # The number of express stations of all express vehicle paths in the global optimal individual
        self.best_dist_list = []  # Global Optimal solution Route Length list (record the distance traveled for each route)
        self.all_best_chrom = []  # Record the change of the global optimal individual during each iteration
        self.all_best_chrom_fit = []  # The fitness changes of the global optimal individual during each iteration were recorded
        self.all_best_dist_sum = []  # Record the total distance of the globally optimal individual during each iteration
        self.cross_rate = 0.8  # cross
        self.mutation_rate = 0.25  # mutatuin
        self.mutation_reverse_rate = 0.7  # "reverse"
        self.cars_driving_distance = []  # driving distance
        self.driving_time=[]  #driving time
        self.distance_weight = distance_weight  # w1
        self.diff_weight=diff_weight  #w2
        self.average_weight=average_weight
        self.time_weight=time_weight #w3

        self.best_time_list=[]
        self.old_route_list=[]

        self.start_index=0

    def get_stations(self):
        express_stations = []
        stations_data_1000 = pd.read_csv('./data/express_station5.csv').values
        m = stations_data_1000[:, 0:1]
        m = [j for i in m for j in i]
        x = stations_data_1000[:, 1:2]
        x = [j for i in x for j in i]
        y = stations_data_1000[:, 2:3]
        y = [j for i in y for j in i]
        for i in range(len(x)):
            express_stations.append([m[i], x[i], y[i]])
        print(express_stations)
        return express_stations

    def get_driving_distance(self):
        cars_driving_distance = []
        cars_data_5 = pd.read_csv('./data/driving_distance_new.csv').values
        x = cars_data_5[:, 1:2]
        x = [j for i in x for j in i]
        cars_driving_distance = x
        print(cars_driving_distance)
        return cars_driving_distance

    def D(slef, location1, location2):  # Calculate the distance between express delivery station Step 1
        return math.sqrt(pow(location1[1] - location2[1], 2) + pow(location1[2] - location2[2], 2))

    def DMAT(self, locations):  # Calculate the distance between express delivery station. Step 2
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

    def express_stations_init_pop(self):
        # Initialize the express_stations
        self.express_stations = self.get_stations()
        self.disMatrix = self.DMAT(self.express_stations)
        self.cars_driving_distance = self.get_driving_distance()
        self.conMatrix=self.get_congestion_situation()
        k_means=K_M(self.express_station_num,self.car_num,self.start_station,self.express_stations,self.cars_driving_distance)
        print("Enter the k-means ++ algorithm stage")
        for i in range(self.population_size):  # population_size is the number of population
            chrom = k_means.k_means_starts()
            self.population.append(chrom)
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

    def cross(self, parent1, parent2, chrom_length):  # 交叉互换
        index1 = np.random.randint(0, chrom_length - 1)  # 交叉初始点 [0，len(parent1) - 1)
        index2 = np.random.randint(index1, chrom_length - 1)  # 交叉结束点[index1，len(parent1) - 1)
        tempGene1 = parent1[index1:index2]  # 交叉的基因片段
        tempGene2 = parent2[index1:index2]
        newGene1, newGene2 = [], []
        p1len, p2len = 0, 0
        for g in parent1:  # 总结：  随机变换+parent2的[index1，index2]+ 随机变换
            if p1len == index1:  # 如果遍历到要交换的初始位置
                newGene1.extend(tempGene2)  # 插入基因片段
            if g not in tempGene2:  # 插入基因片段(一段一段加)往newGene中添加tempGene交叉的基因片段extend方法：启拼接作用，不带[]
                newGene1.append(g)  # 这个是一条一条加 ，append添加 带[]
            p1len += 1

        for g in parent2:  # 遍历899条数据  总结：  随机变换+parent2的[index1，index2]+ 随机变换
            if p2len == index1:  # 如果遍历到要交换的初始位置
                newGene2.extend(tempGene1)  # 插入基因片段
            if g not in tempGene1:  # 插入基因片段(一段一段加)往newGene中添加tempGene交叉的基因片段extend方法：启拼接作用，不带[]
                newGene2.append(g)  # 这个是一条一条加 ，append添加 带[]
            p2len += 1

        return newGene1, newGene2  # 染

    def crossover(self, population, chrom_length, choice):

        new_population = []
        # “杰出选择”选择出新的一代
        selected_pop = self.select_better_pop(population, choice)  # 传入的是种群，返回的是数量相同的种群，但是里面的个体相对优化了
        for i in range(int(self.population_size / 2)):  # -------------------------改：除以2 不然返回的种群数量就变成60了
            two_chrom = random.choices(selected_pop, k=2)  # 随机选取两次
            # 执行OX
            child_chrom1, child_chrom2 = self.cross(two_chrom[0], two_chrom[1], chrom_length)
            new_population.append(child_chrom1)
            new_population.append(child_chrom2)

        return new_population

    def mutate_swap(self, parent_chrom, chrom_length):
        # 如果index1和index2相等，则交换变异相当于没有执行
        index1 = random.randint(self.start_index, chrom_length - 1)
        index2 = random.randint(self.start_index, chrom_length - 1)
        child_chrom = parent_chrom[:]
        child_chrom[index1], child_chrom[index2] = child_chrom[index2], child_chrom[index1]
        return child_chrom

    def mutate_reverse(self, parent_chrom, chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index,
                                                                                            chrom_length - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        child_chrom = parent_chrom[:]
        tmp = child_chrom[index1: index2]
        tmp.reverse()
        child_chrom[index1: index2] = tmp
        return child_chrom


    def mutate_insert(self, parent_chrom, chrom_length):
        index1, index2 = random.randint(self.start_index, chrom_length - 1), random.randint(self.start_index,
                                                                                            chrom_length - 1)
        child_chrom = parent_chrom[:]
        child_chrom.pop(index2)
        child_chrom.insert(index1 + 1, parent_chrom[index2])
        return child_chrom



    def mutation(self, population, chrom_length):
        rate_sum = []
        rate_sum.extend([self.mutation_reverse_rate, 1])
        new_population = []
        for i in range(self.population_size):
            p = random.random()
            if p <= rate_sum[0]:  # 0.4的概率
                # 倒转变异
                child_chrom = self.mutate_reverse(population[i], chrom_length)
                new_population.append(child_chrom)
            else:
                # 插入变异
                child_chrom = self.mutate_insert(population[i], chrom_length)
                new_population.append(child_chrom)
            # else:
            #     # 插入变异
            #     child_chrom = self.mutate_reverse(population[i], chrom_length)
            #     new_population.append(child_chrom)
        return new_population

    def compute_pop_fitness(self, population,choice):
        return [self.fitness_function(chrom,choice) for chrom in population]  # 调用适应度函数

    def get_best_chrom(self, population,choice):
        tmp = self.compute_pop_fitness(population,choice)
        index = tmp.index(max(tmp))
        return population[index], index

    def get_cars_distance(self, chrom):  # 获得各个快递车的行驶距离
        # 起始点5，快递站点9个，快递车3，切断点10,11
        # [4, 6, 11, 9, 2, 1, 10, 7, 8, 3]
        #self.disMatrix=np.array(self.disMatrix)
        tmp_chrom = chrom[:]
        # 将增加的切断点还原成起始点
        for i in range(len(chrom)):
            if chrom[i] in self.break_points:
                tmp_chrom[i] = self.start_station
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个快递车路线，可以为空
        all_routines = []  # 所有快递车路线
        passing_stations_count = []
        for v in tmp_chrom:
            if v == self.start_station:
                all_routines.append(one_routine)
                passing_stations_count.append(len(one_routine))
                one_routine = []
            elif v != self.start_station:
                one_routine.append(v)
        # 还有一次需要添加路线
        all_routines.append(one_routine)
        passing_stations_count.append(len(one_routine))
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
                        distance += self.disMatrix[self.start_station - 1, r[i] - 1]
                    if i + 1 < r_len:
                        #print("%d %d %f" %(r[i] ,r[i+1], self.disMatrix[r[i] - 1, r[i + 1] - 1] ))
                        distance += self.disMatrix[r[i] - 1, r[i + 1] - 1]
                    # 最后一个顶点，下一站是起始点
                    elif i == r_len - 1:
                        distance += self.disMatrix[r[i] - 1, self.start_station - 1]
                routines_dis.append(distance)
        #print(routines_dis)
        return all_routines, routines_dis, passing_stations_count

    def get_cars_distance_and_time(self, chrom):  #新增方法
        length=len(chrom) #这是途径快递站点的数目
        distance = 0.0
        time=0.0
        if(length!=0):
            for i in range(length):
                # 别忘了最后加上起始点到第一个点的距离
                if i == 0:
                    distance += self.disMatrix[self.start_station - 1, chrom[i] - 1]
                    time += (self.disMatrix[self.start_station - 1, chrom[i] - 1] / 5) * self.conMatrix[
                        self.start_station - 1, chrom[i] - 1] #时速算30km/60分钟  ,距离的单位为（百米）例如：30百米=3km需要6分钟，故为(30/10)*2
                if i + 1 < length:
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                    time += (self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1] / 5) * self.conMatrix[
                        chrom[i] - 1, chrom[i + 1] - 1]
                # 最后一个顶点，下一站是起始点
                elif i == length - 1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_station - 1]
                    time += (self.disMatrix[chrom[i] - 1, self.start_station - 1] / 5) * self.conMatrix[
                        chrom[i] - 1, self.start_station - 1]
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
                    distance += self.disMatrix[self.start_station - 1, chrom[i] - 1]
                    time+=(self.disMatrix[self.start_station - 1, chrom[i] - 1]/50) * self.conMatrix[self.start_station - 1, chrom[i] - 1]
                if i + 1 < len(chrom):
                    distance += self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                    time+=(self.disMatrix[chrom[i] - 1, chrom[i + 1] - 1]/50)*self.conMatrix[chrom[i] - 1, chrom[i + 1] - 1]
                # 最后一个顶点，下一站是起始点
                elif i == len(chrom) - 1:
                    distance += self.disMatrix[chrom[i] - 1, self.start_station - 1]
                    time+=(self.disMatrix[chrom[i] - 1, self.start_station - 1]/50) * self.conMatrix[chrom[i] - 1, self.start_station - 1]
            obj2=self.distance_weight * distance + time * self.time_weight
            return obj2


    def fitness_function(self, chrom,choice):
        return math.exp(1.0 / self.obj_function(chrom,choice))

    def ga_process(self):
        self.express_stations_init_pop()  # 初始化种群
        self.ga_process_iterator(self.get_cars_distance)  # 调用GA算法的迭代过程
        print("******************************")
        print(self.max_iteration_num1 + 1)
        print(self.all_best_dist_sum)
        print("******************************")

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
            # 每代最优个体最好的路径组成和每条路路径长度per_gen_best_dist_list以及每条路径途经的站点数self.per_gen_passing_stations
            self.per_gen_best_path, self.per_gen_best_dist_list, self.per_gen_passing_stations = get_distance_func(
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
                self.best_path, self.best_dist_list, self.best_passing_stations = get_distance_func(self.best_chrom)
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
            if self.iteration_count == self.max_iteration_num1:
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
                print(max(self.best_dist_list))
                self.max_iteration_num1 = self.iteration_count
                self.print_iteration()
            elif(self.iteration_count==self.max_iteration_num1-1):
                self.iteration_count+=1
                self.print_iteration()
            # 更新种群
            self.population = pop_new
            # -------------------新的一代有关参数更新结束------------------------------------------------

    def print_iteration(self):
        print("After %d iterations" % self.iteration_count)
        print("The total distance traveled by all EDV:%f,The maximum distance traveled by EDV:%f" % (
        self.best_dis_sum, max(self.best_dist_list)))
        print("The global optimal solution route is{}".format(self.best_path))
        print("Number of passing stations{}".format(self.best_passing_stations))
        print("Global optimal solution route length list{}".format(self.best_dist_list))
        print("---------------------------------------------------------")
        print("The optimal solution distance of each generation：%f,The maximum distance traveled by EDV%f" % (
        self.per_gen_best_dis_sum, max(self.per_gen_best_dist_list)))
        print("The optimal solution route of each generation{}".format(self.per_gen_best_path))
        print("Number of passing stations{}".format(self.per_gen_passing_stations))
        print("Each generation optimal solution route length list{}".format(self.per_gen_best_dist_list))
        print("**************************************************************************")

    def print_best_routine(self):
        """
        打印快递车最优航线
        Returns:
        """
        print(type(self.best_path))
        print("EDVs' all route length：{}".format(self.best_dis_sum))
        #best_path, best_dist_list, passing_stations_count = self.get_cars_distance(self.best_chrom)
        # 打印全局最优个体的所有路线快递站点（包括起点和终点）
        for i in range(len(self.best_path)):
            print("he route length of {} EDV is {}".format(i + 1, self.best_dist_list[i]))
            print("The route of {} EDV：".format(i + 1), end="")
            if len(self.best_path[i]) == 0:
                print("The EDV does not leave")  # 这种情况可以通过设置目标函数避免
            else:
                for j in range(len(self.best_path[i])):
                    if j == 0:
                        print("{} ——> {} ".format(int(self.express_stations[self.start_station - 1][0]),
                                                  int(self.express_stations[self.best_path[i][j] - 1][0])), end="")
                    if j + 1 < len(self.best_path[i]):
                        print("——> {} ".format(int(self.express_stations[self.best_path[i][j + 1] - 1][0])), end="")
                    elif j == len(self.best_path[i]) - 1:
                        print("——> {}".format(int(self.express_stations[self.start_station - 1][0])))

if __name__ == "__main__":
    # 快递
    start = time.time()
    ga_obj = GA(car_num=5, express_station_num=1001, start_station=1001,max_iteration_num1=1000, max_iteration_num2=5000,
                distance_weight=1,diff_weight=1000,average_weight=100,time_weight=100)
    ga_obj.ga_process()
    end = time.time()
    print("Time consuming:", end - start)
#-----tradition GA  --stage1
#After 333 iterations The total distance traveled by all EDV:9929.255053,The maximum distance traveled by EDV:2369.695031
#After 159 iterations The total distance traveled by all EDV:9899.147013,The maximum distance traveled by EDV:2253.325192
#After 143 iterations The total distance traveled by all EDV:9830.705805,The maximum distance traveled by EDV:2239.431216
#After 52 iterations The total distance traveled by all EDV:9683.605466,The maximum distance traveled by EDV:2392.851455
#After 356 iterations The total distance traveled by all EDV:9855.458827,The maximum distance traveled by EDV:2382.019637
#After 5 iterations The total distance traveled by all EDV:9923.656603,The maximum distance traveled by EDV:2309.108141
#After 492 iterations The total distance traveled by all EDV:9740.144604,The maximum distance traveled by EDV:2298.314923
#After 104 iterations The total distance traveled by all EDV:9922.412071,The maximum distance traveled by EDV:2395.472447
#After 178 iterations The total distance traveled by all EDV:9965.541954,The maximum distance traveled by EDV:2352.571527
#After 180 iterations The total distance traveled by all EDV:9973.256733,The maximum distance traveled by EDV:2274.929464
#After 594 iterations The total distance traveled by all EDV:9972.891414,The maximum distance traveled by EDV:2307.898686


#----improved GA  ---stage1
#After 1 iterationsThe total distance traveled by all EDV:9802.328683,The maximum distance traveled by EDV:2232.491561
#After 56 iterationsThe total distance traveled by all EDV:9829.953458,The maximum distance traveled by EDV:2388.596524
#After 126 iterations The total distance traveled by all EDV:9647.795817,The maximum distance traveled by EDV:2396.398119
#After 1 iterationsThe total distance traveled by all EDV:9970.031394,The maximum distance traveled by EDV:2269.697581
#After 47 iterations The total distance traveled by all EDV:9888.602082,The maximum distance traveled by EDV:2395.940352
#After 19 iterations The total distance traveled by all EDV:9855.402790,The maximum distance traveled by EDV:2329.342422
#After 239 iterations The total distance traveled by all EDV:9446.657905,The maximum distance traveled by EDV:2259.196608
#After 62 iterations The total distance traveled by all EDV:9716.706848,The maximum distance traveled by EDV:2325.286639
#After 50 iterations The total distance traveled by all EDV:9836.109673,The maximum distance traveled by EDV:2331.709944
#After 164 iterations The total distance traveled by all EDV:9412.978995,The maximum distance traveled by EDV:2169.542320

