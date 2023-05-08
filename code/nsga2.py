import math
import random
import pandas as pd
import matplotlib.pyplot as plt

'''
# Function to find index of list

def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1

# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list

# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                    values1[p] <= values1[q] and values2[p] < values2[q]) or (
                    values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                    values1[q] <= values1[p] and values2[q] < values2[p]) or (
                    values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
                # print("n[p]", n[p])
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# Function to calculate crowding distance


def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    return distance


# Function to carry out the crossover


def crossover(a, b, min_x, max_x):
    r = random.random()
    if r > 0.5:
        return mutation((a + b) / 2, min_x, max_x)
    else:
        return mutation((a - b) / 2, min_x, max_x)


# Function to carry out the mutation operator
def mutation(solution, min_x, max_x):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * random.random()
    return solution


# First function to optimize


def func_comp(x):
    return x


# Second function to optimize


def func_mem(x):
    return x

def func_band(x):
    return x


# Main program starts here


def nsga_2(df):
    pop_size = 100
    max_gen = 5

    # Initialization
    min_x = -55
    max_x = 55
    solution_comp = df['proc'].to_list()
    solution_mem = df['mem'].to_list()
    solution_band = df['band'].to_list()
    # print(solution_fcmi)
    # print(solution_affmi)
    # press_key = input("press any key to continue ")
    gen_no = 0
    while gen_no < max_gen:
        function1_values = [func_comp(solution_comp[i]) for i in range(0, pop_size)]
        # print(function1_values)

        function2_values = [func_mem(solution_mem[i]) for i in range(0, pop_size)]
        # print(function2_values)
        function3_values = [func_band(solution_band[i]) for i in range(0, pop_size)]
        # print(function2_values)
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print(non_dominated_sorted_solution)
        print("The best front for Generation number ", gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print(round(solution_comp[valuez], 3), end=" ")
            print(round(solution_mem[valuez], 3), end=" ")
            print(round(solution_band[valuez], 3), end=" ")
        print("\n")
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
        solution2 = solution_comp[:]
        # Generating offsprings
        while len(solution2) != 2 * pop_size:
            a1 = random.randint(0, pop_size - 1)
            b1 = random.randint(0, pop_size - 1)
            solution2.append(crossover(solution_fcmi[a1], solution_affmi[b1], min_x, max_x))
        function1_values2 = [func_comp(solution2[i]) for i in range(0, 2 * pop_size)]
        function2_values2 = [func_mem(solution2[i]) for i in range(0, 2 * pop_size)]
        function3_values3 = [func_band()]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == pop_size:
                    break
            if len(new_solution) == pop_size:
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    # Lets plot the final front now
    print("Proc :", function1_values)
    print("MR :", function2_values)
    # df1 = pd.DataFrame({'Proc': function1_values, 'MR': function2_values})
    # print("df1",df1)
    # df2 = pd.merge(df, df1['Proc'], on=[''], how='inner')
    # df2.drop(df2.index[df2['FCMI'] == 0.0], inplace=True)
    # print("Client list", df2)
    # function1 = [i * 1 for i in function1_values]
    # function2 = [j * 1 for j in function2_values]
    plt.xlabel('Maximize Processing capacity', fontsize=15)
    plt.ylabel('Maximize MR', fontsize=15)
    plt.scatter(function1_values, function2_values)
    plt.show()

# return df2

"""Server receive 
meta data of Memory availability and Processing capacity from clients"""
def MOOCS_nsga2(proc, mem, comm):
    proc = [11, 10, 82, 19, 35, 79, 39, 9, 87, 88, 10, 62, 85, 64, 99, 53, 42, 61, 12, 47, 71, 93, 3, 71, \
            34, 59, 57, 13, 36, 3, 36, 69, 94, 35, 44, 30, 86, 20, 80, 33, 97, 16, 45, 64, 84, 23, 90, 53, \
            43, 37, 42, 95, 86, 51, 86, 17, 46, 32, 1, 1, 74, 18, 89, 99, 99, 65, 25, 67, 23, 29, 27, 94, 43, \
            65, 13, 1, 59, 85, 34, 25, 49, 0, 26, 7, 11, 13, 21, 70, 67, 20, 83, 20, 20, 75, 49, 71, 4, 15, 19, 2]

    proc1 =  [ -x1 for x1 in proc]
    mem = [29, 28, 18, 5, 9, 36, 47, 46, 30, 29, 41, 34, 7 , 6, 48, 7, 40, 35, 30, 29, 49, 8, 14, 22, 8, 7, 3, \
            34, 34, 22, 25, 28, 5, 4, 6, 13, 43, 26, 19, 38, 37, 2, 32, 9, 40, 34, 20, 50, 20, 15, 32, 37, 4, \
            45, 39, 11, 49, 31, 2, 48, 31, 47, 44, 6, 40, 20, 21, 43, 7, 49, 7, 18, 49, 31, 7, 50, 11, 22, 45, \
            18, 31, 36, 20, 24, 41, 19, 48, 13, 5, 33, 38, 34, 9, 21, 45, 2, 44, 27, 39, 4]

    mem1 =  [ -x1 for x1 in mem]
     df = pd.DataFrame(list(zip(proc1, mem1)),
                   columns =['proc', 'mem'])
    #print(df)
    df.to_csv(r'df_to_data.txt', header=None, index=None, sep=' ', mode='a')
    nsga_2(df)


'''





