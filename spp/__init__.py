from .problem import Instance, load_instance, is_feasible, solution_cost
from .simulated_annealing import simulated_annealing
from .ga_standard import binary_genetic_algorithm
from .ga_improved import improved_genetic_algorithm

ALGORITHMS = {
    "sa": ("Simulated Annealing", simulated_annealing),
    "bga": ("Binary Genetic Algorithm", binary_genetic_algorithm),
    "ibga": ("Improved Binary Genetic Algorithm", improved_genetic_algorithm),
}
