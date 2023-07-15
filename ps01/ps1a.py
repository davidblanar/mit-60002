###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    result = {}
    with open(filename) as f:
        for line in f.readlines():
            data = line.strip().split(',')
            name, weight = data[0], int(data[1])
            result[name] = weight
    return result

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    sorted_cows = sorted([(value, key) for key, value in cows.items() if value <= limit], key=lambda x: x[0], reverse=True)
    trips = []
    while len(sorted_cows) > 0:
        sublist = []
        index = 0
        current_trip_weight = 0
        while True:
            if index == len(sorted_cows):
                trips.append(sublist)
                break
            weight, name = sorted_cows[index]
            if current_trip_weight + weight <= limit:
                sublist.append(name)
                del sorted_cows[index]
                current_trip_weight += weight
            else:
                index += 1
    return trips

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    def are_trips_viable(trips, limit):
        is_viable = True
        for trip in trips:
            total = 0
            for cow in trip:
                total += cows[cow]
            if total > limit:
                is_viable = False
        return is_viable

    generator = get_partitions(cows)
    shortest_len_seen = limit
    best_found = []
    for trips in generator:
        if are_trips_viable(trips, limit):
            if len(trips) < shortest_len_seen:
                shortest_len_seen = len(trips)
                best_found = trips
    return best_found

# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    cows = load_cows('./ps01/ps1_cow_data.txt')

    start_1 = time.time()
    solution_1 = brute_force_cow_transport(cows)
    end_1 = time.time()
    print(f'Brute force found a solution with {len(solution_1)} trips in {(end_1 - start_1):.6f} seconds')

    start_2 = time.time()
    solution_2 = greedy_cow_transport(cows)
    end_2 = time.time()
    print(f'Greedy approach found a solution with {len(solution_2)} trips in {(end_2 - start_2):.6f} seconds')


if __name__ == '__main__':
    assert greedy_cow_transport({"Jesse": 11}) == []
    assert greedy_cow_transport({"Jesse": 10}) == [['Jesse']]
    assert greedy_cow_transport({"Jesse": 4}) == [['Jesse']]
    assert greedy_cow_transport({"Jesse": 6, 'Maybel': 5}) == [['Jesse'], ['Maybel']]
    assert greedy_cow_transport({"Jesse": 5, 'Maybel': 5}) == [['Jesse', 'Maybel']]
    assert greedy_cow_transport({"Jesse": 5, 'Maybel': 3, 'Callie': 2, 'Maggie': 5}) == [
        ['Jesse', 'Maggie'], ['Maybel', 'Callie'],
    ]
    cows = load_cows('./ps01/ps1_cow_data.txt')
    assert greedy_cow_transport(cows) == [
        ['Betsy'],
        ['Henrietta'],
        ['Herman', 'Maggie'],
        ['Oreo', 'Moo Moo'],
        ['Millie', 'Milkshake', 'Lola'],
        ['Florence']
    ]
    cows = load_cows('./ps01/ps1_cow_data_2.txt')
    assert greedy_cow_transport(cows) == [
        ['Lotus'],
        ['Horns'],
        ['Dottie', 'Milkshake'],
        ['Betsy', 'Miss Moo-dy', 'Miss Bella'],
        ['Rose']
    ]

    compare_cow_transport_algorithms()
