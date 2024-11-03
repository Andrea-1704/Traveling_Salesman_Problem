# CI2024_lab2

In this lab, I solved the Traveling Salesman Problem (TSP) by using a greedy algorithm and a genetic algorithm. 

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
For a testing porpouse I also considered scramble mutation and inversion mutation, but avoided them.

For the crossover, I used inver_over_crossover and inversion_crossover.
For a testing porpouse I also considered pmx crossover and order crossover.

### Parent selection
For the parent selection I used tournament selection where tau, after a fine tuning process, is put as population_size/8.

### Initial population
For the initial population I considered the solution provided by the greedy algorithm for a random starting city and then used __simulated annealing__. 
I also tried considering "random solution" (also combined with the greedy ones), but this did not improve the results, so I decided to avoid it.

### EA approach
I tried all the possibles EA approach and selected the __hyper modern approach__ because provided better results.

### Parameter selection
Since, sometimes the algorithm did not reach the best solution in a limited amount of time (sometimes also after high values for the generations number), the idea was to change the number of generations according to the problem size: in some dataset is better to continue running the same instance for a bigger number of generations. In others, instead, is better to execute more than once different instances of the same algorithm to reach to an optimum solution (as an improvement would be better to do it in parallel using threads).


## Results (EA)

|  Instance     | Distance(Km) |
|---------------|--------------|
| Vanuatu       | 1.345,54     |
| Italy         | 4.172,76     |
| Russia        | 32.574       |
| US            | 40.204       |
| China         | 55.804       |

I avoided to put the digits after comma when the distance was already high.

These results are obtained using a number of generations that depends on the size of the problem.

### Other informations

The code is computationally intensive to complete. Feel free to reduce the number of generations if you need a less time-consuming approach. Throughout the project, I collaborated with Stefano Fumero, consulting with him on various implementation decisions.