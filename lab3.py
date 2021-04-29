# import random function
import random

# import numpy for working with numbers
import numpy as np

# used to create a chromosome and take it as the reference solution
def create_reference_solution(chromosome_length):
    # number of 1s is equal to number of 0s
    number_of_ones = int(chromosome_length / 2)
    
    # build an array with all 0s
    reference = np.zeros(chromosome_length)
    
    # from index 1 to the number of 1s, count the number of elements,
        # and change all these elements to 1
    reference[0: number_of_ones] = 1
    
    # randomly shuffle the array to mix the 0s and 1s
    np.random.shuffle(reference)
 
    # returns the array for chromosome as the result
    return reference

# print an example target array as a 70-gene reference chromosome 
print (create_reference_solution(70))

# used to create a population with individuals (number of individual chromosomes)
def create_starting_population(individuals, chromosome_length):
    
    # set up an initial array of all 0s
    population = np.zeros((individuals, chromosome_length))
    
    # loop through rows/individuals
    for i in range(individuals):
        
        # choose a random number of ones to create
        ones = random.randint(0, chromosome_length)
        
        # change this required number of 0s to 1s
        population[i, 0:ones] = 1
        
        # shuffle rows
        np.random.shuffle(population[i])
 
    # returns the array for population as the result
    return population

# print a random population of 4 individuals with gene length of 10
print (create_starting_population(4, 10))

# used to evaluate the fitness by matching the genes (elements) in a 
    # potential solution (chromosome) against that of the reference standard
def calculate_fitness(reference, population):
    
    # create an array of True/False as compared to reference, for each element
    identical_to_reference = population == reference
    
    # sum the total number of elements that are same as the reference
    fitness_scores = identical_to_reference.sum(axis=1) # 1 is true
 
    # return the array containing the scores of each solution
    return fitness_scores

# print an example target array as a 10-gene reference chromosome 
reference = create_reference_solution(10)
print ('Reference solution: \n', reference)

# print a random population of 6 individuals with gene length of 10
population = create_starting_population(6, 10)
print ('\nStarting population: \n', population)

# evaluate the fitness, and print the scores
scores = calculate_fitness(reference, population)
print('\nScores: \n', scores)

# used to choose individuals for breeding
def select_individual_by_tournament(population, scores):
    
    # find the population size
    population_size = len(scores)
 
    # choose 2 random individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)
 
    # find the fitness score for each individual
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]
 
    # identify individual with higher fitness. Fighter 1 will win if equal scores 
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
 
    # Return the chromosome of the winner. Winner is the index.
    return population[winner, :]

# 10-gene reference chromosome as reference
reference = create_reference_solution(10)

# random population of 6 individuals with gene length of 10
population = create_starting_population(6, 10)

# evaluate fitness and score population
scores = calculate_fitness(reference, population)

# choose two parents and display
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)
print (parent_1)
print (parent_2)

# used to perform crossover and produce children
def breed_by_crossover(parent_1, parent_2):
    
    # find total length of chromosome
    chromosome_length = len(parent_1)
 
    # randomly choose and cut at the crossover point, avoding ends of chromsome
    crossover_point = random.randint(1,chromosome_length-1)
 
    # produce children with mix of genes from each parent
    # np.hstack joins 2 smaller arrays into 1 big array
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))
    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))
 
    # return children
    return child_1, child_2

# 15-gene reference chromosome as reference
reference = create_reference_solution(15)

# random population of 100 individuals with gene length of 15
population = create_starting_population(100, 15)

# evaluate fitness and score population
scores = calculate_fitness(reference, population)

# choose two parents
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)

# produce children through crossover
child_1, child_2 = breed_by_crossover(parent_1, parent_2)

# display output
print ('Parents')
print (parent_1)
print (parent_2)
print ('Children')
print (child_1)
print (child_2)

# used to randomly mutate genes
def randomly_mutate_population(population, mutation_probability):
 
    # random size to mutate
    random_mutation_array = np.random.random(
        size=(population.shape))
 
    # check that random size selected is within mutation probability
    random_mutation_boolean = \
        random_mutation_array <= mutation_probability
        
    # random selection to randomly convert the 0s and 1s
    population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])
 
    # return the mutation population
    return population

# 15-gene reference chromosome as reference
reference = create_reference_solution(15)

# random population of 100 individuals with gene length of 15
population = create_starting_population(100, 15)

# evaluate fitness and score population
scores = calculate_fitness(reference, population)

# choose two parents
parent_1 = select_individual_by_tournament(population, scores)
parent_2 = select_individual_by_tournament(population, scores)

# produce children through crossover
child_1, child_2 = breed_by_crossover(parent_1, parent_2)

# add children to new population 
population = np.stack((child_1, child_2))
print ("Population before mutation")
print (population)

# mutation rate
mutation_probability = 0.25

# display population after random mutation
population = randomly_mutate_population(population, mutation_probability)
print ("Population after mutation")
print (population)



# set the general parameters

# length of chromosome
chromosome_length = 75

# size of population
population_size = 500

# continue the loop until this specified maximum generation is reached
maximum_generation = 200

# track progress
best_score_progress = []

# create reference solution 
reference = create_reference_solution(chromosome_length)

# create starting population with individuals
population = create_starting_population(population_size, chromosome_length)

# evaluate and record scores in starting population
scores = calculate_fitness(reference, population)

# compute the best score of every population, by dividing the number of True
    # by the total length
best_score = np.max(scores)/chromosome_length * 100
print ('Starting best score, percent target: %.1f' %best_score)

# add the starting best score to progress tracker
best_score_progress.append(best_score)

# continue through generations of GA until 200
for generation in range(maximum_generation):
    
    # create an empty list for new population
    new_population = []
    
    # create new population by producing two children at a time through crossover
    for i in range(int(population_size/2)):
        
        # choose two parents
        parent_1 = select_individual_by_tournament(population, scores)
        parent_2 = select_individual_by_tournament(population, scores)
        
        # produce children through crossover
        child_1, child_2 = breed_by_crossover(parent_1, parent_2)
        
        # add children to new population 
        new_population.append(child_1)
        new_population.append(child_2)
 
    # replace the old population with the new population. 
    # New take over from the old.
    population = np.array(new_population)
 
    # mutation rate
    mutation_rate = 0.002
    
    # randomly mutate population
    population = randomly_mutate_population(population, mutation_rate)
    
    # evaluate and record scores
    scores = calculate_fitness(reference, population)
    
    # compute the best score
    best_score = np.max(scores)/chromosome_length * 100
    
    # records the best score of every generation in the progress tracker
    best_score_progress.append(best_score)
    
# GA has completed all required generations
print ('End best score, percent target: %.1f' %best_score)

# plot the progress
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# progress tracker
plt.plot(best_score_progress)

# x axis is the generations
plt.xlabel('Generation')

# y axis is the best scores
plt.ylabel('Best score (% target)')
plt.show()