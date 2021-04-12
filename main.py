import numpy as np
import random 
import matplotlib.pyplot as plt
from tqdm import trange
def to_binary(x, k = 0):
    x_bin_tmp = bin(x)
    x_binary = x_bin_tmp[2:]
    if k == 0:
        k = len(x_binary)
    x_binary_prefix = '0' * (k - len(x_binary)) # Create an empty string with lenght of k
    x_binary = x_binary_prefix + x_binary
    return x_binary


def select_pair_wheel(pmf):
    x = random.uniform(0,1)
    #select mother
    min_v = 0
    for i in range(0, len(pmf)):
        max_v =  pmf[i] + min_v
        #print("x: %.2f Interval [%.2f -- %.2f |" %(x,min_v, max_v  ))  
        if x > min_v and  x  <= max_v:
            mother_idx = i
        min_v = max_v

    #select father
    x = random.uniform(0,1)
    min_v = 0
    for i in range(0, len(pmf)):
        max_v =  pmf[i] + min_v
        #print("x: %.2f Interval [%.2f -- %.2f |" %(x,min_v, max_v  ))  
        if x > min_v and  x  <= max_v:
            father_idx = i
        min_v = max_v
    return mother_idx, father_idx




def mutation(binary_number = "0000", prob_mutation = 0.05):
    binary_number = list(binary_number)
    for i in range(len(binary_number)-1, -1, -1):
        prob = random.uniform(0, 1)
        if prob < prob_mutation:
            if binary_number[i] == "0":
                binary_number[i] = "1"
            else:
               binary_number[i] = "0"
            break   
    return "".join(binary_number)


def crossover_v2(binary_number_1 = "0000", binary_number_2 = "1111"):
    binary_number_1 = list(binary_number_1)
    binary_number_2 = list(binary_number_2)
    for i in range(0, len(binary_number_1)):
        prob= random.uniform(0,1)
        if prob < 0.5:
            tmp = binary_number_1[i]
            binary_number_1[i] = binary_number_2[i]
            binary_number_2[i] = tmp
    binary_number_1 = "".join(binary_number_1)
    binary_number_2 = "".join(binary_number_2)

    return binary_number_1, binary_number_2


def crossover(binary_number_1 = "0000", binary_number_2 = "1111"):
    loci = [i for i in range(1, len(binary_number_1))]
    position_to_cut = int(np.floor(len(binary_number_1)*0.5))
    binary_number_1 = list(binary_number_1)
    binary_number_2 = list(binary_number_2)
    #first crossover
    for i in range(position_to_cut, len(binary_number_1)):
        tmp_char = binary_number_1[i]
        binary_number_1[i] = binary_number_2[i]
        binary_number_2[i] = tmp_char



    binary_number_1 = "".join(binary_number_1)
    binary_number_2 = "".join(binary_number_2)

    return binary_number_1, binary_number_2

def create_individual(size_of_it):

    individual_x = list(range(0, size_of_it))
    individual_x_dec = list(range(0, 17))
    for i in range(0, len(individual_x)):
        prob = np.random.uniform(0,1)
        if prob >= 0.5:
            individual_x[i] = "1"
        else:
            individual_x[i] = "0"

    for i in range(0, len(individual_x_dec)):
        prob = np.random.uniform(0,1)
        if prob >= 0.5:
            individual_x_dec[i] = "1"
        else:
            individual_x_dec[i] = "0"


    individual_y = list(range(0, size_of_it))
    individual_y_dec = list(range(0, 17))

    for i in range(0, len(individual_y)):
        prob = np.random.uniform(0,1)
        if prob >= 0.5:
            individual_y[i] = "1"
        else:
            individual_y[i] = "0"

    for i in range(0, len(individual_y_dec)):
        prob = np.random.uniform(0,1)
        if prob >= 0.5:
            individual_y_dec[i] = "1"
        else:
            individual_y_dec[i] = "0"


    individual = []
    individual.append("".join(individual_x))                    
    individual.append("".join(individual_x_dec))                    

    individual.append("".join(individual_y))                    
    individual.append("".join(individual_y_dec))                    

    return individual


def create_population(size_of_pop, each_individual_binary_size):
    population = []
    for i in range(0, size_of_pop):
        population.append(create_individual(each_individual_binary_size))
    return population

def get_fitnnes(population):
    fit_population = []
    for i in range(0, len(population)):
        x = int(population[i][0][1:],2)
        if population[i][0][0] == '1':
            x = -x

        x_dec = int(population[i][1],2)*1e-6

        x = x + x_dec
        y = int(population[i][2][1:],2)
        if population[i][2][0] == '1':
            y = -y
        y_dec = int(population[i][3],2)*1e-6
        y = y + y_dec

     

        tmp = x**2 + y**2
        den = ((1 + 0.001*(tmp)))**2
        num = np.sin(np.sqrt(tmp))**2 - 0.5
        z = 0.5 - num/den
        fit_population.append(z)
 
 
    pmf = fit_population/np.sum(fit_population)
    best_fitness = np.max(fit_population)
    return  fit_population, pmf, best_fitness



def create_new_generation_v2(population, pmf):
    new_generation = []

    #for i in range(0, int(0.5*len(population))):
    for i in trange(int(0.5*len(population))):
        #print(i)
        mother_idx, father_idx = select_pair_wheel(pmf)
        mother = population[mother_idx]
        father = population[father_idx]

        child_1_x, child_2_x = crossover(mother[0], father[0])
        child_1_x_dec, child_2_x_dec = crossover(mother[1], father[1])

        child_1_y, child_2_y = crossover(mother[2], father[2])
        child_1_y_dec, child_2_y_dec = crossover(mother[3], father[3])


        child_1_x = mutation(child_1_x, prob_mutation = 0.05)
        child_1_y = mutation(child_1_y, prob_mutation = 0.05)
        
        child_1_x_dec = mutation(child_1_x_dec, prob_mutation = 0.05)
        child_1_y_dec = mutation(child_1_y_dec, prob_mutation = 0.05)

        child_2_x = mutation(child_2_x, prob_mutation = 0.05)
        child_2_y = mutation(child_2_y, prob_mutation = 0.05)

        child_2_x_dec = mutation(child_2_x_dec, prob_mutation = 0.05)
        child_2_y_dec = mutation(child_2_y_dec, prob_mutation = 0.05)



        child_1 = [child_1_x, child_1_x_dec, child_1_y, child_1_y_dec]
        child_2 = [child_2_x, child_2_x_dec, child_2_y, child_2_y_dec]
        new_individual = []
        new_generation.append(child_1)
        new_generation.append(child_2)

    return new_generation

def create_new_generation(selected_individuals):
    current_pop = selected_individuals
    new_generation = []
    flag = True
    while flag:
        #select the pair index
        select_parent_index = random.sample(range(0,len(current_pop)),2)
        
        #collect each pair individual from population
        pair_1 = current_pop[select_parent_index[0]]
        pair_2 = current_pop[select_parent_index[1]]

        #crossover step
        cross_1, cross_2 = crossover(pair_1, pair_2)

        new_generation.append(mutation(cross_1, prob_mutation=0.1))
        new_generation.append(mutation(cross_2, prob_mutation=0.1))

        #mutation step

        #remove the already selected members
        current_pop.pop(select_parent_index[0])
        current_pop.pop(select_parent_index[1]-1)
        flag = False if not current_pop else True
    return new_generation

def selection_step(population, fit_population):
    tuples = list(zip(population, fit_population))
    sorted_tuples = sorted(tuples, key=lambda student: student[1])   # sort by age
    selected_individuals_tuple = sorted_tuples[int(0.8*np.floor(len(population))):]
    selected_individuals = []
    for i in range(0, len(selected_individuals_tuple)):
        selected_individuals.append(selected_individuals_tuple[i][0])

    return selected_individuals



def ga(it):
    generations = 100
    plot_values = []
    best_value_int_x_all = []
    best_value_int_y_all = []
    z_all = []
    population = create_population(size_of_pop = 15000, each_individual_binary_size = 8)    

    max_value = -10000
    for i in range(0, generations):
        fit_population, pmf, best_fitness = get_fitnnes(population)
        #selected_individuals = selection_step(population, fit_population)
        index_best = np.where(pmf == np.max(pmf))[0][0]
        
        best_value_bin_x = population[index_best][0]
        best_value_bin_x_decimal = population[index_best][1]

        best_value_bin_y = population[index_best][2]
        best_value_bin_y_decimal = population[index_best][3]


        best_value_int_x = int(best_value_bin_x[1:], 2)
        best_value_int_x_dec = int(best_value_bin_x_decimal, 2)*1e-6

        if best_value_bin_x[0] == '1':
            best_value_int_x = -best_value_int_x
        best_value_int_x += best_value_int_x_dec


        best_value_int_y = int(best_value_bin_y[1:], 2)
        best_value_int_y_dec = int(best_value_bin_y_decimal, 2)*1e-6

        if best_value_bin_y[0] == '1':
            best_value_int_y = -best_value_int_y
        best_value_int_y += best_value_int_y_dec
        
        
        print("iteration: %d | best value (x,y): (%.6f, %.6f): %.6f| generation: %d | best_value: %.5f" %(it, best_value_int_x, best_value_int_y , best_fitness,  i, max_value))
        best_value_int_x_all.append(best_value_int_x)
        best_value_int_y_all.append(best_value_int_y)
        z_all.append(best_fitness)
        
        if best_fitness > max_value:
            max_value = best_fitness
            best_x_all = best_value_int_x
            best_y_all = best_value_int_y


        plot_values.append(best_fitness)
        population = create_new_generation_v2(population, pmf)

    import pickle
    pickle.dump([best_value_int_x_all, best_value_int_y_all, z_all], open("results_iteration_" + str(it) + ".pickle", "wb"))

    plt.plot(plot_values)
    plt.title("max_value = %.6f" %(best_fitness))
    plt.xlabel("Generation")
    plt.ylabel("Evolution of the best value")
    #plt.show()
  
        #new_generation = create_new_generation(selected_individuals)
    #plt.show() 
    #print("===========================================")
    #print("best (x,y): (%d, %d) | best fit: %.6f" %(best_x_all, best_y_all, max_value))   
    return best_x_all, best_y_all, max_value



def main():
    best_x_all = -1000
    best_y_all = -1000
    max_value_all = -1000
    best_it = -1
    for it in range(0,10):
        print("==================================")
        print("==================================")

        best_x, best_y, max_value = ga(it)
        if max_value > max_value_all:
            best_x_all = best_x
            best_y_all = best_y
            max_value_all = max_value
            best_it = it
        #print("===========================================")
        #print("best (x,y): (%d, %d) | best fit: %.6f" %(best_x, best_y, max_value))   
        #print("===========================================")
        print("==================================")
        print("==================================")

        plt.savefig("results_it_"+str(it)+".png", format="png")
        plt.close()
    print("====== Simulation ended ======")
    print("(x,y) = (%6f, %6f)" %(best_x_all, best_y_all))
    print("z = %.6f" %(max_value_all))
    print("best iteration: %d" %(best_it))
if __name__ == "__main__":
  main()
