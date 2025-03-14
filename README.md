# Travelling Salesman Problem

In this project, I solved the Travelling Salesman Problem (TSP) by using a greedy algorithm and a genetic algorithm. 

The problem involves a salesman who needs to visit a set of cities, each exactly once, and return to the starting city. The goal is to find the shortest possible route that covers all cities and minimizes the travel distance (or cost, depending on the context).


# Implementation

## Greedy approach
For the greedy approach I computed the distances between each couple of cities in the following way:

```python
DIST_MATRIX = np.zeros((len(CITIES), len(CITIES)))
for c1, c2 in combinations(CITIES.itertuples(), 2):
    DIST_MATRIX[c1.Index, c2.Index] = DIST_MATRIX[c2.Index, c1.Index] = geodesic(
        (c1.lat, c1.lon), (c2.lat, c2.lon)
    ).km

segments = [
        ({c1, c2}, float(DIST_MATRIX[c1, c2])) for c1, c2 in combinations(range(len(CITIES)), 2)
]
```

The greedy algorithm simply starts from a starting city, and continue taking the nearest city to the current one, until all the cities are reached.

A snapshot of the code that performs that operation is here provided:

```python
def greedy_sol(city, segments):
    solution = []
    solution.append(city)
    visited = []
    visited.append(int(city))
    while len(visited)<len(CITIES):
        _, c1 = find_closest(segments, city, visited)
        solution.append(c1)
        visited.append(c1)
        city=c1
    solution.append(solution[0])
    
    return solution
```

To perform a greedy algorithm for a given dataset we can consider all the possible starting city and choose the one which minimize the total distance. 
However, this would lead to an increase of the time required for the computation, so I decided to simply start from a __random city__.



## Evolutionary algorithm

### Fitness function
I defined the fitness function as the opposite of the total distance of a path:

```python
def fitness(solution):
    tot_dist=0
    for node in range(len(solution)-1):
        tot_dist -= DIST_MATRIX[solution[node], solution[node+1]]
    return tot_dist
```

All the transformations used by the algorithm do not "invalidate" a solution, meaning that are designed to not violate the constraints of the problem (if we start from a valid solution), so there's no need to add other informations to the fitness function.

### Mutation and crossover
I considered the swap mutation and his variant (called in the code "mutation_strength"). These mutation functions provided the best solutions among the others.
For a testing porpouse I also considered scramble mutation and inversion mutation, but avoided them because they provided worse results.

```python
def swap_mutation(solution):
    index = random.randint(1, len(solution)-3) #not the last one nor the first
    index2=index #should be higher
    while index2<=index:
        index2 = random.randint(1, len(solution)-2)
    selected_edge1 = solution[index]
    selected_edge2 = solution[index2]
    solution[index] = selected_edge2
    solution[index2] = selected_edge1
    return solution
```

```python
def mutate_strength(individual, mutationRate=0.5):
    for swapped in range(1,len(individual)-1):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual)-1)
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual
```

For the crossover, I used inversion_crossover.
For a testing porpouse I also considered pmx crossover and order crossover, but didn't use them due to their lack in performances.

```python
def crossover_inversion(seq1, seq2):
    dim = len(seq1) - 1  
    pos1, pos2 = sorted(random.sample(range(dim), 2))

    inverted_segment = seq1[pos1:pos2 + 1][::-1] 
    new_son = [None] * dim
    new_son[pos1:pos2 + 1] = inverted_segment

    indice_seq2 = 0
    for k in range(dim):
        if new_son[k] is None:
            while seq2[indice_seq2] in new_son:
                indice_seq2 += 1
            new_son[k] = seq2[indice_seq2]

    new_son.append(new_son[0])
    return new_son
```

### Parent selection
For the parent selection I used tournament selection where tau, after a fine tuning process, is put as population_size/8.

```python
def parent_selection(population):
    candidates = sorted(np.random.choice(population, int(len(population)/8)), key=lambda e: e.fitness, reverse=True)
    return candidates[0]
```

Is value of tau provided a good selective pressure, meaining a good balance between diversity and preserving the best traits.

### Initial population
For the initial population I considered the solution provided by the greedy algorithm for a random starting city and then used __simulated annealing__. 

```python
def simulated_annealing(solution):
    initial_temperature = 100
    heating_rate = 1.02
    iteration = 0

    # One out of five approach
    recent_improvements = deque(maxlen=5)
    required_improvements = 1  

    # Stopping criteria:
    recent_improvements_stop = deque(maxlen=1000)
    recent_improvements_stop.append(True)

    # Initial solution: greedy one!
    current_solution = solution
    current_cost = fitness(current_solution)
    best_cost = current_cost
    best_solution = current_solution

    temperature = initial_temperature
    while iteration < 1_000:
        iteration += 1
        # Tweak the solution
        random_number = random.random()
        first_time = True
        while random_number > 0.8 or first_time:
            first_time = False
            new_solution = swap_mutation(current_solution.copy())
            new_cost = fitness(new_solution)
            random_number = random.random()
        
        # Variation of fitness by changing sign
        cost_delta = new_cost * (-1) - current_cost * (-1)
        # We are sure the solution after swap mutation is valid if the previous one was.

        if cost_delta < 0 or (random.random() < math.exp(-cost_delta / temperature) and cost_delta != 0):
            current_solution = new_solution
            current_cost = new_cost
            recent_improvements.append(True)
            recent_improvements_stop.append(True)
            if current_cost * (-1) < best_cost * (-1):
                best_cost = current_cost
                best_solution = current_solution
        else:
            recent_improvements.append(False)
            recent_improvements_stop.append(False)

        if recent_improvements.count(True) > required_improvements:
            temperature *= heating_rate  # More exploration
        if recent_improvements.count(True) < required_improvements:
            temperature /= heating_rate

        if recent_improvements_stop.count(True) == 0:  # Stop condition
            break
    return best_solution
```

I also tried considering "random solution" (also combined with the greedy ones), but this did not improve the results, so I decided to avoid it. I also tried using an initial population given by using 2 different greedy algorithms but did not improve the performances.

### EA approach
I tried all the possibles EA approach and selected the __hyper modern approach__ because provided better results.

```python
def execute_EA(small_db = False, POPULATION_SIZE=100, OFFSPRING_SIZE=200, MAX_GENERATIONS=200):
    segments = [
        ({c1, c2}, float(DIST_MATRIX[c1, c2])) for c1, c2 in combinations(range(len(CITIES)), 2)
    ]
    population = [Individual(simulated_annealing(greedy_sol(random.randint(0, len(CITIES)-1), segments))) for _ in range(int(POPULATION_SIZE))]
    # while len(population)<POPULATION_SIZE:
    #     population.append(Individual(genome=simulated_annealing(create_random_solution())))
    for i in population:
        i.fitness = fitness(i.genome)
    population.sort(key=lambda i: i.fitness, reverse=True)


    for g in range(MAX_GENERATIONS):
        offspring = []
        for _ in range (OFFSPRING_SIZE):
            if np.random.random()<0.4:#mutation probability:
                p=parent_selection(population)
                if small_db:
                    o=swap_mutation(p.genome.copy())
                else:
                    o=mutate_strength(p.genome.copy())
            else:
                i1 = parent_selection(population)
                i2 = parent_selection(population)
                o = crossover_inversion(i1.genome.copy(), i2.genome.copy())
            offspring.append(Individual(genome=o, fitness =fitness(o)))
            if small_db:
                o2 = swap_mutation(o.copy())
            else:
                o2 = mutate_strength(o.copy())
            if np.random.random()<0.05:
                o3 = simulated_annealing(o2.copy())
                offspring.append(Individual(genome=o3, fitness =fitness(o3)))
            offspring.append(Individual(genome=o2, fitness =fitness(o2)))

        population.extend(offspring)
        population.sort(key=lambda i: i.fitness, reverse=True)
        population = population[:POPULATION_SIZE]
        if g%50==0:
            print("sol so far at gen: ", g, " is: ",fitness(population[0].genome)*(-1))

    population.sort(key=lambda i: i.fitness, reverse=True)
    population = population[:POPULATION_SIZE]
    return population[0]

best_fitness = float('inf')*(-1)
best_sol = None

instances0 = [(100, 200, 20)]
instances1 = [(100, 305, 200), (100, 305, 200), (50, 80, 300)]
instances2 = [(100, 305, 1_500)]
instances3 = [(100, 305, 2_450)]

best_fitness = float('-inf')
best_sol = None

if len(CITIES)<100:
    instances=instances1
if len(CITIES)<30:
    instances=instances0
if len(CITIES)>100:
    instances=instances2
if len(CITIES)>200:
    instances=instances3


for current in instances:
    if len(CITIES)<50:
        valore = execute_EA(True,current[0], current[1], current[2]) 
    else:
        valore = execute_EA(False,current[0], current[1], current[2]) 
    if valore.fitness > best_fitness:
        best_fitness = valore.fitness
        best_sol = valore.genome

print("best fitness: ", best_fitness*(-1))
print(best_sol)
```


I fine-tuned all the meta-parameters (number of generations, population size, and offspring size) for each problem instance and dataset considered.

Since the algorithm occasionally failed to reach the optimal solution within a limited time (even with a high number of generations), I adjusted the number of generations based on the problem size. For some datasets, it proved more effective to run the algorithm for a greater number of generations. For others, running multiple independent instances of the algorithm was more beneficial in achieving an optimal solution. As a future improvement, these independent runs could be executed in parallel using threads for greater efficiency.

The function also includes a parameter to indicate whether the dataset is small, as I observed that swap mutation provided better performance in shorter timeframes for these instances.


### Parameter selection
This solution aims to enhance population diversity by using a large offspring size and generating new generations through crossover and mutation applied to the initial population. Producing 200 offspring ensures broader exploration of the search space in each generation. The high offspring count introduces a variety of genetic variations into the population, significantly reducing the risk of the algorithm stagnating in local optima.

The population size is set to 100, which is an unconventional choice but was determined to be optimal through a fine-tuning process, yielding the best performance.

This decision is based on the observation that a smaller population size (100) allows computational resources to focus on evaluating fewer parent solutions. Meanwhile, the larger offspring pool (200) ensures the introduction of diverse genetic material into subsequent generations.

With a limited population size, the algorithm retains only the top-performing individuals, ensuring steady progress toward better solutions. These elite individuals guide the search in subsequent generations, while the large offspring pool facilitates exploration of the solution space.


## Results (EA)

|  Instance     | Distance(Km) |
|---------------|--------------|
| Vanuatu       | 1.345,54     |
| Italy         | 4.172,76     |
| Russia        | 35.574       |
| US            | 40.204       |
| China         | 55.804       |

I avoided to put the digits after comma when the distance was already high.

These results are obtained using a number of generations that depends on the size of the problem.

I avoid considering also the execution time since for this algorithm there was not a clear metric for this, and I would like to avoid considering machine execution time, because it varies a lot depending on the computer used.

### Conclusions

The code is computationally intensive to complete. Feel free to reduce the number of generations if you need a less time-consuming approach. Throughout the project, I collaborated with Stefano Fumero, consulting with him on various implementation decisions.


## After reviews
After the review received by Professor Giuseppe Esposito I followed his suggestion to be consistent with the English language, also for in-line comments.
