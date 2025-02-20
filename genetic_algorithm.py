import random, math, os
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ---------------------------
# Generate 20 Cities (including the depot)
# ---------------------------
def generate_cities(num_cities):
    cities = []
    depot = {"id": 0, "x": 50, "y": 50}  # Depot at center
    cities.append(depot)
    for i in range(1, num_cities):
        city = {"id": i, "x": random.uniform(0, 100), "y": random.uniform(0, 100)}
        cities.append(city)
    return cities


# ---------------------------
# Helper Functions
# ---------------------------
def euclidean_distance(city1, city2):
    return math.hypot(city1["x"] - city2["x"], city1["y"] - city2["y"])


def evaluate_route(route, cities):
    total_distance = 0.0
    current_city = cities[0]  # start at depot
    for city_idx in route:
        next_city = cities[city_idx]
        total_distance += euclidean_distance(current_city, next_city)
        current_city = next_city
    total_distance += euclidean_distance(current_city, cities[0])
    return total_distance


# ---------------------------
# Genetic Algorithm Operators
# ---------------------------
def create_individual(num_cities):
    individual = list(range(1, num_cities))  # Exclude depot
    random.shuffle(individual)
    return individual


def tournament_selection(population, cities, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: evaluate_route(ind, cities))
    return tournament[0]


def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end + 1] = parent1[start:end + 1]
    pos = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
    return child


def swap_mutation(individual, mutation_rate=0.1):
    individual = individual[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# ---------------------------
# Plotting Function for the TSP Route
# ---------------------------
def plot_route(route, cities, title="TSP Route"):
    full_route = [0] + route + [0]
    xs = [cities[i]["x"] for i in full_route]
    ys = [cities[i]["y"] for i in full_route]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker='o', linestyle='-', color='b')
    plt.scatter(xs, ys, color='red')
    for i in full_route:
        plt.text(cities[i]["x"] + 1, cities[i]["y"] + 1, str(cities[i]["id"]),
                 fontsize=10, color='green')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.grid(True)


# ---------------------------
# Main GA Loop with Frame Saving and Cost Recording
# ---------------------------
def genetic_algorithm(cities, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2):
    population = [create_individual(len(cities)) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')
    cost_history = []  # Record best cost at each generation

    # Directory for saving route screenshots
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    frame_files = []  # list to store frame file paths

    for gen in range(generations):
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]  # clone parent1
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population

        # Update the best solution and record the cost
        for individual in population:
            fitness = evaluate_route(individual, cities)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = individual
        cost_history.append(best_fitness)

        # Every 10 generations, plot and save the current best route.
        if gen % 10 == 0:
            title = f"Generation {gen} - Best Distance: {best_fitness:.2f}"
            print(title)
            plot_route(best_solution, cities, title=title)
            filename = os.path.join(frames_dir, f"frame_{gen}.png")
            plt.savefig(filename)
            plt.close()  # Free up memory
            frame_files.append(filename)

    # ---------------------------
    # Plot Cost vs. Generation Graph
    # ---------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(range(generations), cost_history, marker='o', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("Total Distance")
    plt.title("Evolution of Best Cost over Generations")
    plt.grid(True)
    cost_graph_filename = "distance_vs_generation.png"
    plt.savefig(cost_graph_filename)
    plt.close()
    print(f"Cost vs. Generation graph saved as {cost_graph_filename}")

    return best_solution, best_fitness, frame_files


# ---------------------------
# Run the GA and Create a GIF from Screenshots
# ---------------------------
if __name__ == "__main__":
    NUM_CITIES = 20
    cities = generate_cities(NUM_CITIES)

    best_route, best_route_fitness, frame_files = genetic_algorithm(cities)
    best_route_full = [0] + best_route + [0]
    print("\nFinal Best Route (including depot start/end):", best_route_full)
    print("Final Best Total Distance:", best_route_fitness)

    # Create GIF from saved frames
    gif_filename = "tsp_evolution.gif"
    frames = []
    for filename in frame_files:
        frames.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, frames, duration=0.5)
    print(f"GIF saved as {gif_filename}")
