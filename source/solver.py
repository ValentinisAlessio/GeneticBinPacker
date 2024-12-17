import numpy as np
from tqdm import trange

class GeneticBinPacker():

    def __init__(self, 
                 population_size:   int,
                 mutation_rate:     float,
                 crossover_rate:    float,
                 generations:       int,
                 k_elite:           int,
                 tournament_size:   int,
                 buff_size:         int     = 10,
                 set_seed:          bool    = False):
        
        # Set seed
        if set_seed:
            np.random.seed(42)

        # Algorithm parameters
        self.columns_per_individual     = None
        self.bricks_per_column          = None
        self.population_size            = population_size
        self.mutation_rate              = mutation_rate
        self.crossover_rate             = crossover_rate
        self.generations                = generations
        self.k_elite                    = k_elite
        self.tournament_size            = tournament_size
        self.population                 = None
        self.bricks                     = None                  # Attrbute containing the brick heights to use in each individual

        # Attributes to store the best individual and fitness, along with the history of the algorithm
        self.best_individual            = None                  
        self.best_fitness               = np.inf
        self.history                    = []

        # For early stopping
        self.delta                      = np.inf
        self.buff_size                  = buff_size
        self.delta_buff                 = []
    

    def column_height(self, column):
        '''
        Method to compute the height of a column

        Parameters
        -------------------
            - column (np.array):
                The column to compute the height

        Returns
        -------------------
            - int: The height of the column
        '''
        return np.sum(column)
    

    def is_odd(self, number):
        '''
        Method to check if a number is odd using bitwise operations for efficiency
        
        Parameters
        -------------------
            - number (int):
                The number to check
                
        Returns
        -------------------
            - bool: True if the number is odd, False otherwise
        '''

        return number & 1
    

    def evaluate_fitness(self, individual):
        '''
        Method to evaluate the fitness of an individual as the difference between the maximum and minimum height of the columns
        
        Parameters
        -------------------
            - individual (np.array):
                The individual to evaluate the fitness
                
        Returns
        -------------------
            - int: The fitness of the individual
        '''
        
        min_height, max_height = np.sort([self.column_height(column) for column in individual])[[0, -1]]

        return max_height - min_height
    
    
    def evaluate_fitness_pop(self):
        '''
        Method to evaluate the fitness of the entire population

        Returns
        -------------------
            - np.array: The fitness of each individual in the population
        '''

        # Compute the height of each column of the entire population
        column_heights  = np.sum(self.population, axis=2)

        # Compute the min and max height of each individual
        min_heights     = np.min(column_heights, axis=1)
        max_heights     = np.max(column_heights, axis=1)

        return max_heights - min_heights
    
    
    def evaluate_fitness_pop_normalized(self):
        '''
        Method to evaluate the fitness of the entire population normalized by the mean height of the bricks.
        Used to compare the fitness of different populations with different brick distributions

        Returns
        -------------------
            - np.array: The fitness of each individual in the population normalized by the mean height of the bricks
        '''

        return self.evaluate_fitness_pop() / np.mean(self.bricks)
    

    def generate_bricks(self, gen_type='random'):
        '''
        Method to generate the bricks to use in the individuals

        Parameters
        -------------------
            - gen_type (str):
                The type of generation to use. Can be 'random', 'uniform' or 'normal'
                * 'random': brick heights are taken from a random permutation of the numbers from 1 to the number of bricks
                * 'uniform': brick heights are taken from a uniform distribution between 0 and 10
                * 'normal': brick heights are taken from a normal distribution with mean 5 and std 2
        '''

        num_bricks = self.columns_per_individual * self.bricks_per_column

        if      gen_type =='random':
            self.bricks = np.random.permutation(np.arange(1, num_bricks + 1))
        elif    gen_type =='uniform':
            self.bricks = np.random.uniform(0, 10, size=num_bricks)
        elif    gen_type =='normal':
            self.bricks = np.random.normal(5, 2, size=num_bricks)
            self.bricks = np.clip(self.bricks, 0.1, np.inf)   # Avoid negative values, very unlikely but yet possible


    def generate_individual(self):
        '''
        Method to generate a single individual  

        Returns
        -------------------
            - np.array: The generated individual
        '''

        # Take a random permutation of the bricks
        bricks      = np.random.permutation(self.bricks)

        # Reshape into the desired shape
        individual  = bricks.reshape(self.columns_per_individual, self.bricks_per_column)

        return individual
    
    
    def initialize_population(self):
        '''
        Method to initialize the population

        Returns
        -------------------
            - np.array: The initialized population
        '''

        init_population = [self.generate_individual() for _ in range(self.population_size)]
        return np.array(init_population)
    

    def check_individual(self, individual):
        '''
        Method to check if an individual is valid, i.e. if all the bricks are different
        
        Parameters
        -------------------
            - individual (np.array):
                The individual to check
                
        Returns
        -------------------
            - bool: True if the individual is valid, False otherwise
        '''

        bricks = []
        for column in individual:
            for brick in column:
                if brick in bricks:
                    return False
                bricks.append(brick)

        return True
    
    
    def elitism(self, normalize=False):
        '''
        Method that implements elitism to select the best individuals of the population
        
        Parameters
        -------------------
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
                
        Returns
        -------------------
            - np.array: The best individuals of the population
        '''

        # Solve the case when the population size xor elitism size are odd
        if self.is_odd(self.population_size) ^ self.is_odd(self.k_elite):
            self.k_elite    += 1

        # Take the fitness of the population
        if normalize:
            fitness_scores  = self.evaluate_fitness_pop_normalized()
        else:
            fitness_scores  = self.evaluate_fitness_pop()

        # Sort the population by fitness
        sorted_idx          = np.argsort(fitness_scores)
        self.population     = self.population[sorted_idx]

        # Take the (k_elite) best individuals
        elites              = self.population[:self.k_elite].copy()
        self.population     = self.population[self.k_elite:].copy()

        # Shuffle the rest of the population lose the ordering
        np.random.shuffle(self.population)

        return elites
    
    
    def tournament_selection(self, normalize=False):
        '''
        Method to perform tournament selection to select the best individual from a subset of the population.

        Parameters
        -------------------
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
        
        Returns
        -------------------
            - np.array: The best individual selected by the tournament
        '''

        # Take the indexes of the individuals to use in the tournament
        tournament_idx      = np.random.choice(self.population.shape[0], size=self.tournament_size, replace=False)
        tournament          = self.population[tournament_idx]

        # Take the fitness of the tournament individuals
        if normalize:
            fitness_scores  = self.evaluate_fitness_pop_normalized()[tournament_idx]
        else:
            fitness_scores  = self.evaluate_fitness_pop()[tournament_idx]

        # Take the best individual
        best_idx            = np.argmin(fitness_scores)

        return tournament[best_idx]
    

    def crossover(self, 
                  parent1: np.array, 
                  parent2: np.array, 
                  method: str ='pmx'):
        '''
        Method to perform the crossover between two parents to generate two children
        
        Parameters
        -------------------
            - parent1 (np.array):
                The first parent to use in the crossover
            - parent2 (np.array):
                The second parent to use in the crossover
            - method (str):
                The method to use in the crossover. Can be 'pmx' or 'cycle'
                * 'pmx': Partially Mapped Crossover
                * 'cycle': Cycle Crossover
        
        Returns
        -------------------
            - np.array: The first child generated by the crossover
            - np.array: The second child generated by the crossover
        '''

        # Check the validity of the inputs
        if not isinstance(method, str):
            raise ValueError("Crossover method must be a string of type 'pmx' or 'cycle'")
        if method not in ['pmx', 'cycle']:
            raise ValueError("Crossover method must be a string of type 'pmx' or 'cycle'")
        
        # Perform the crossover
        if method == 'pmx':
            child1, child2 = self.pmx_crossover(parent1, parent2)

        elif method == 'cycle':
            child1, child2 = self.cycle_crossover(parent1, parent2)


        return child1, child2
    

    def pmx_crossover(self, 
                      parent1: np.array, 
                      parent2: np.array):
        '''
        Method to perform the Partially Mapped Crossover (PMX) between two parents to generate two children.

        Parameters
        -------------------
            - parent1 (np.array):
                The first parent to use in the crossover
            - parent2 (np.array):
                The second parent to use in the crossover
        
        Returns
        -------------------
            - np.array: The first child generated by the crossover
            - np.array: The second child generated by the crossover
        '''

        random_idx  = np.random.randint(0, parent1.shape[0])
        column1     = parent1[random_idx].copy()
        column2     = parent2[random_idx].copy()


        # !WARNING! The mapping created here does not preserve the order of the swaps, i.e.
        #       | 1 |          | 4 |
        #       | 2 |    ->    | 5 |
        #       | 3 |          | 6 |
        # does NOT create the mapping {1: 4, 2: 5, 3: 6}.
        #
        # However, in this problem this is not foundamental, as the columns
        #       | 1 |          | 3 |
        #       | 2 |    &     | 1 | 
        #       | 3 |          | 2 |
        # are equivalent in terms of height.


        # Generate the mapping to swap the bricks
        x   = list(set(column1) - set(column2))
        y   = list(set(column2) - set(column1))
        # This way the mapping avoids duplicates.

        map = dict(zip(x, y))

        # Create the children
        child1  = parent1.copy()
        child2  = parent2.copy()

        # Permute the elements following the brick mapping
        for key, value in map.items():
            child1 = np.where(child1 == value, key, child1)
            child2 = np.where(child2 == key, value, child2)

        # Swap the columns
        child1[random_idx]  = column2
        child2[random_idx]  = column1

        return child1, child2
    
    
    def cycle_crossover(self, 
                        parent1: np.array, 
                        parent2: np.array):
        '''
        Method to perform the Cycle Crossover between two parents to generate two children.

        Parameters
        -------------------
            - parent1 (np.array):
                The first parent to use in the crossover
            - parent2 (np.array):
                The second parent to use in the crossover

        Returns
        -------------------
            - np.array: The first child generated by the crossover
            - np.array: The second child generated by the crossover
        '''

        # Select a random index to start the cycle
        idx     = np.random.randint(0, parent1.shape, (2,))

        # Create the children
        child1  = parent1.copy()
        child2  = parent2.copy()

        # Create a mask to keep track of the visited elements
        visited = np.zeros(parent1.shape, dtype=bool)

        # Perform the cycle crossover
        while not visited[*idx]:
            visited[*idx]   = True	
            child2          = np.where(parent1 == parent2[*idx], parent2[*idx], child2)
            child1          = np.where(parent2 == parent1[*idx], parent1[*idx], child1)
            idx             = np.where(parent2 == parent1[*idx])


        return child1, child2


    def mutation(self, 
                 individual: np.array):
        '''
        Method to perform the mutation of an individual by swapping two random bricks in two random columns
        
        Parameters
        -------------------
            - individual (np.array):
                The individual to mutate
        
        Returns
        -------------------
            - np.array: The mutated individual
        '''

        # Select the columns and bricks to swap
        mask = np.random.randint(0, individual.shape, size=(2, 2))

        # Swap the bricks
        individual[*mask[0]], individual[*mask[1]]  = individual[*mask[1]], individual[*mask[0]]

        return individual
    
    
    def generate_new_population(self, 
                                crossover: str, 
                                normalize: bool = False):
        '''
        Method to execute the main loop of the Genetic Algorithm, performing elitism, selection, crossover and mutation
        
        Parameters
        -------------------
            - crossover (str):
                The method to use in the crossover. Can be 'pmx' or 'cycle'
                * 'pmx': Partially Mapped Crossover
                * 'cycle': Cycle Crossover
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
        '''

        # Initialize the new population
        new_population  = np.empty((0, *self.population.shape[1:]))

        # Elitism step
        new_population  = np.concatenate((new_population, self.elitism(normalize=normalize)), axis=0)

        # Selection step
        selected_pop    = np.array([self.tournament_selection(normalize=normalize) for _ in range(self.population.shape[0])])

        # Generate the pairs of parents to crossover as half of the population
        selected_idx    = np.array([
                            np.random.choice(selected_pop.shape[0], size=2, replace=False) for _ in range(self.population.shape[0] // 2)
                        ])

        # Crossover and mutation step
        children = []

        for i, j in selected_idx:
            parent1, parent2    = selected_pop[i], selected_pop[j]

            # This way I ensure that if neither crossover nor mutation are performed, the children are the same as the parents
            # and will be added to the new population
            child1, child2      = parent1.copy(), parent2.copy()

            # Crossover
            if np.random.rand() < self.crossover_rate:
                child1, child2  = self.crossover(parent1, parent2, method=crossover)
            
            # Mutation
            if np.random.rand() < self.mutation_rate:
                child1          = self.mutation(child1)
            if np.random.rand() < self.mutation_rate:
                child2          = self.mutation(child2)
            
            children.append(child1)
            children.append(child2)

        new_population  = np.concatenate((new_population, children), axis=0)

        self.population = np.array(new_population)


    def update_best(self, 
                    iteration: int, 
                    termination_criterion: str = None, 
                    normalize: bool = False):
        '''
        Method to update the best individual and fitness of the solver

        Parameters
        -------------------
            - iteration (int):
                The current iteration of the algorithm
            - termination_criterion (str):
                The criterion to use for early stopping. Can be 'delta' or 'std'
                * 'delta': stop with a probability proortional to the fitness improvement
                * 'std': stop with probability proportional to the standard deviation of the fitness of the population
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
        '''

        # Store the last best fitness
        last_best   = self.best_fitness

        # Evaluate the fitness of the population
        if normalize:
            fitness_scores = self.evaluate_fitness_pop_normalized()
        else:
            fitness_scores = self.evaluate_fitness_pop()

        # Take the best individual and fitness
        best_idx                = np.argmin(fitness_scores)
        self.best_individual    = self.population[best_idx].copy()
        self.best_fitness       = fitness_scores[best_idx]

        # Enter this part only if a termination criterion is specified
        if termination_criterion is not None:

            if      termination_criterion == 'delta':
                self.delta = last_best - self.best_fitness if self.best_fitness < last_best else self.delta

            elif    termination_criterion == 'std':
                self.delta = np.std(fitness_scores)

        # Store the record of the iteration in the history
        record = {
            'iteration'         : iteration,
            'population'        : self.population,
            'best_individual'   : self.best_individual,
            'best_fitness'      : self.best_fitness
        }

        self.history.append(record)


    def update_delta_buff(self, 
                          delta: float):
        '''
        Method to update the delta buffer used in the dynamic stopping criterion
        
        Parameters
        -------------------
            - delta (float):
                The delta to add to the buffer
        '''
        # Append the delta to the buffer
        self.delta_buff.append(delta)

        # Keep the buffer size constant by deleting the oldest element
        if len(self.delta_buff) > self.buff_size:
            self.delta_buff.pop(0)


    def compute_T(self, 
                  k: float):
        '''
        Method to compute the temperature T for the dynamic stopping criterion
        
        Parameters
        -------------------
            - k (float):
                The scaling factor to use in the computation of T
        
        Returns
        -------------------
            - float: The computed temperature T
        '''

        # If the buffer is empty, return 1
        if not self.delta_buff or len(self.delta_buff) == 1:
            return 1
    
        # Compute the standard deviation of the buffer and scale it by k
        return np.std(self.delta_buff) * k

    def solve(self, 
              columns_per_individual: int, 
              bricks_per_column: int, 
              brick_dist: str = 'random', 
              crossover: str = 'pmx', 
              normalize: bool = False, 
              verbose: bool = False):
        '''
        Method to solve the Bin Packing problem using a Genetic Algorithm
        
        Parameters
        -------------------
            - columns_per_individual (int):
                The number of columns in each individual
            - bricks_per_column (int):
                The number of bricks in each column
            - brick_dist (str):
                The type of distribution to use for the bricks. Can be 'random', 'uniform' or 'normal'
                * 'random': brick heights are taken from a random permutation of the numbers from 1 to the number of bricks
                * 'uniform': brick heights are taken from a uniform distribution between 0 and 10
                * 'normal': brick heights are taken from a normal distribution with mean 5 and std 2
            - crossover (str):
                The method to use in the crossover. Can be 'pmx' or 'cycle'
                * 'pmx': Partially Mapped Crossover
                * 'cycle': Cycle Crossover
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
            - verbose (bool):
                If True, a progress bar is shown during the execution of the algorithm to show the progress
        '''

        # Set the columns per individual and bricks per column attributes
        self.columns_per_individual     = columns_per_individual
        self.bricks_per_column          = bricks_per_column

        # Generate the bricks
        self.generate_bricks(gen_type = brick_dist)

        # Initialize the population
        self.population                 = self.initialize_population()
        self.update_best(0, normalize=normalize)

        # Main loop of the algorithm
        if verbose:
            for gen in trange(self.generations):
                self.generate_new_population(crossover=crossover, normalize=normalize)
                self.update_best(gen + 1, normalize=normalize)
        else:
            for gen in range(self.generations):
                self.generate_new_population(crossover=crossover, normalize=normalize)
                self.update_best(gen + 1, normalize=normalize)

    
    def solve_with_termination(self, 
                               columns_per_individual: int, 
                               bricks_per_column: int, 
                               termination_criterion: str, 
                               T: float, 
                               brick_dist: str = 'random', 
                               crossover: str = 'pmx',
                               verbose: bool = False, 
                               normalize: bool = False):
        '''
        Method to solve the Bin Packing problem using a Genetic Algorithm with a termination criterion.
        Stops with probability P_stop = exp(-delta / T)

        Parameters
        -------------------
            - columns_per_individual (int):
                The number of columns in each individual
            - bricks_per_column (int):
                The number of bricks in each column
            - termination_criterion (str):
                The criterion to use for early stopping. Can be 'delta' or 'std'
                * 'delta': stop with a probability proortional to the fitness improvement
                * 'std': stop with probability proportional to the standard deviation of the fitness of the population
            - T (float):
                The temperature to use for scaling the stopping criterion
            - brick_dist (str):
                The type of distribution to use for the bricks. Can be 'random', 'uniform' or 'normal'
                * 'random': brick heights are taken from a random permutation of the numbers from 1 to the number of bricks
                * 'uniform': brick heights are taken from a uniform distribution between 0 and 10
                * 'normal': brick heights are taken from a normal distribution with mean 5 and std 2
            - crossover (str):
                The method to use in the crossover. Can be 'pmx' or 'cycle'
                * 'pmx': Partially Mapped Crossover
                * 'cycle': Cycle Crossover
            - verbose (bool):
                If True, a progress bar is shown during the execution of the algorithm to show the progress
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
        '''

        # Set the columns per individual and bricks per column attributes
        self.columns_per_individual = columns_per_individual
        self.bricks_per_column      = bricks_per_column

        # Generate the bricks
        self.generate_bricks(gen_type = brick_dist)

        # Initialize the population
        self.population     = self.initialize_population()
        self.update_best(0, normalize=normalize)

        iteration           = 1
        continue_evolution  = True

        # Main loop of the algorithm
        if verbose:
            with trange(self.generations) as t:
                while continue_evolution:
                    # Compute the stopping probability
                    p_stop      = np.exp(-self.delta / T)
                    
                    t.set_description(f"Iteration {iteration}, p_stop={p_stop:.4f}")

                    # Check termination conditions
                    if iteration == self.generations or np.random.rand() < p_stop:
                        continue_evolution  = False
                        print(f"Stopping at iteration {iteration}")

                    # Perform evolution steps
                    self.generate_new_population(crossover=crossover, normalize=normalize)
                    self.update_best(iteration, termination_criterion=termination_criterion, normalize=normalize)
                    
                    # Increment iteration and tqdm bar
                    iteration  += 1
                    t.update(1)

        else:
            while continue_evolution:
                # Compute the stopping probability
                p_stop      = np.exp(-self.delta / T)

                # Check termination conditions
                if iteration == self.generations or np.random.rand() < p_stop:
                    continue_evolution = False
                    print(f"Stopping at iteration {iteration}")
                
                # Perform evolution steps
                self.generate_new_population(crossover=crossover, normalize=normalize)
                self.update_best(iteration, termination_criterion=termination_criterion, normalize=normalize)

                # Increment iteration
                iteration  += 1


    def solve_dynamic(self, 
                      columns_per_individual: int, 
                      bricks_per_column: int, 
                      termination_criterion: str, 
                      k: float = 0.1,
                      brick_dist: str = 'random', 
                      crossover: str = 'pmx', 
                      normalize: bool = False, 
                      verbose: bool = False):
        '''
        Method to solve the Bin Packing problem using a Genetic Algorithm with a dynamic stopping criterion.
        Stops with probability P_stop = exp(-delta / T), with dynamic T computed as the standard deviation of the delta buffer

        Parameters
        -------------------
            - columns_per_individual (int):
                The number of columns in each individual
            - bricks_per_column (int):
                The number of bricks in each column
            - termination_criterion (str):
                The criterion to use for early stopping. Can be 'delta' or 'std'
                * 'delta': stop with a probability proortional to the fitness improvement
                * 'std': stop with probability proportional to the standard deviation of the fitness of the population
            - k (float):
                The scaling factor to use in the computation of T
            - brick_dist (str):
                The type of distribution to use for the bricks. Can be 'random', 'uniform' or 'normal'
                * 'random': brick heights are taken from a random permutation of the numbers from 1 to the number of bricks
                * 'uniform': brick heights are taken from a uniform distribution between 0 and 10
                * 'normal': brick heights are taken from a normal distribution with mean 5 and std 2
            - crossover (str):
                The method to use in the crossover. Can be 'pmx' or 'cycle'
                * 'pmx': Partially Mapped Crossover
                * 'cycle': Cycle Crossover
            - normalize (bool):
                If True, the normalized fitness is used to select the best individuals, otherwise the raw fitness is used
            - verbose (bool):
                If True, a progress bar is shown during the execution of the algorithm to show the progress
        '''

        # Set the columns per individual and bricks per column attributes
        self.columns_per_individual = columns_per_individual
        self.bricks_per_column      = bricks_per_column

        # Generate the bricks
        self.generate_bricks(gen_type=brick_dist)

        # Initialize the population
        self.population     = self.initialize_population()
        self.update_best(0, normalize=normalize)

        iteration           = 1
        continue_evolution  = True

        # Main loop of the algorithm
        if verbose:
            with trange(self.generations) as t:
                while continue_evolution:
                    # Compute dynamic T
                    T       = self.compute_T(k)
                    p_stop  = np.exp(-self.delta / T)

                    t.set_description(f"Iteration {iteration}, p_stop={p_stop:.4f}, T={T:.4f}")

                    # Check stopping conditions
                    if iteration == self.generations or np.random.rand() < p_stop:
                        continue_evolution = False

                    # Evolution steps
                    self.generate_new_population(crossover=crossover, normalize=normalize)
                    self.update_best(iteration, termination_criterion=termination_criterion, normalize=normalize)

                    # Update delta buffer
                    self.update_delta_buff(self.delta)

                    # Increment iteration and tqdm bar
                    iteration += 1
                    t.update(1)
        else:
            while continue_evolution:
                # Compute dynamic T
                T       = self.compute_T(k)
                p_stop  = np.exp(-self.delta / T)

                # Check stopping conditions
                if iteration == self.generations or np.random.rand() < p_stop:
                    continue_evolution = False
                    print(f"Stopping at iteration {iteration} with T={T:.4f}")

                # Evolution steps
                self.generate_new_population(crossover=crossover, normalize=normalize)
                self.update_best(iteration, termination_criterion=termination_criterion, normalize=normalize)

                # Update delta buffer
                self.update_delta_buff(self.delta)

                # Update iteration
                iteration += 1

    # Import the plotter functions
    from .plotter import plot_individual, plot_best_individual, plot_population, plot_history