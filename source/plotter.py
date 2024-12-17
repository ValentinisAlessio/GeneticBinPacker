import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from ipywidgets import interact

def plot_individual(self, 
                    individual: np.array, 
                    ax: plt.Axes, 
                    print_values: bool = False
                    ):
    '''
    Plot an individual as a stacked bar chart.
    
    Parameters:
    -------------------
        - individual: 2D numpy array representing the individual to plot.
        - ax: Matplotlib axes object where to plot the individual.
        - print_values: Boolean indicating whether to print the values of each brick in the individual.
    '''

    for column in range(individual.shape[0]):

        col_height = 0                              # Height of the current column

        for brick in range(individual.shape[1]):

            ax.bar(column, individual[column][brick], bottom=col_height, color=cm.jet(individual[column][brick]/np.max(self.bricks)))
            col_height += individual[column][brick]

            if print_values:

                ax.text(column, col_height-individual[column][brick]/2, f"{individual[column][brick]:5.1f}", ha='center', va='center')

        ax.text(column, col_height, f"{col_height:5.1f}", ha='center', va='bottom', fontweight='bold')
        

def plot_best_individual(self, 
                         print_values: bool = False,
                         save_path: str = None):
    '''
    Plot the best individual found by the genetic algorithm.

    Parameters:
    -------------------
        - print_values: Boolean indicating whether to print the values of each brick in the individual.
        - save_path: Path where to save the plot. If None, the plot will be displayed in the notebook.
    '''

    # Normalize the colormap
    norm = mcolors.Normalize(vmin=np.min(self.bricks), vmax=np.max(self.bricks))

    fig, ax = plt.subplots(figsize=(8, 6))
    self.plot_individual(self.best_individual, ax, print_values)
    ax.set_title(f"Best individual | Fitness: {self.best_fitness:.2f}")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Height")
    ax.set_xlim(-0.5, self.columns_per_individual - 0.5)
    ax.set_ylim(0, np.max([self.column_height(column) for column in self.best_individual]) * 1.1)
    plt.colorbar(cm.ScalarMappable(cmap=cm.jet, norm=norm), ax=ax)

    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def plot_population(self):
    '''
    ! WARNING ! Use this function only for small populations.
    Plot all the history of the genetic algorithm.
    '''

    def print_results(generation):

        _, pop, _, _ = self.history[generation].values()
        pop_size = len(pop)

        fig, ax = plt.subplots((pop_size//4 + (1 if pop_size%4 else 0)), 4, figsize=(20, 4 * (pop_size//4 + (1 if pop_size%4 else 0))))
        ax = ax.flatten()
        fig.suptitle(f"Generation {generation}")

        for i, individual in enumerate(pop):
            ax[i].set_title(f"Individual {i}, Fitness: {self.evaluate_fitness(individual):.1f}")
            self.plot_individual(individual, ax[i])

        plt.show()

    # Create the slider
    interact(print_results, generation=widgets.IntSlider(min=0, max=len(self.history)-1, step=1, value=1))

def plot_history(self, path: str = None):
    '''
    Plot the history of the genetic algorithm as a GIF of best individuals.

    Parameters:
    -------------------
        - path: Path where to save the GIF. If None, the GIF will be displayed in the notebook.
    '''

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        # Clear the plot
        ax.clear()
        # Get the best individual and fitness from history for the current frame
        record = self.history[frame]
        best_individual = record['best_individual']
        best_fitness = record['best_fitness']

        # Plot the best individual for the current frame
        self.plot_individual(best_individual, ax)
        # Add titles and labels
        ax.set_title(f"Generation {record['iteration']} | Fitness: {best_fitness}")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Height")
        ax.set_xlim(-0.5, self.columns_per_individual - 0.5)
        ax.set_ylim(0, np.max([self.column_height(column) for column in best_individual]) * 1.1)

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(self.history), repeat=False)

    # Save as GIF
    anim.save(f"{path}genetic_bin_packer.gif", writer="imagemagick", fps=10)

    plt.show()
