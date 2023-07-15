# 6.0002 Problem Set 5
# Graph optimization
# Name:
# Collaborators:
# Time:

#
# Finding shortest paths through MIT buildings
#
import unittest
from graph import Digraph, Node, WeightedEdge

#
# Problem 2: Building up the Campus Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer:
# The nodes represent the different buildings on MIT campus.
# The edges are the pathways between those buildings.
# The distances are represented on the edges - the total distance and the outdoor distance.


# Problem 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """
    print("Loading map from file...")
    g = Digraph()
    with open(map_filename) as f:
        for l in f.readlines():
            line_arr = l.strip().split(' ')
            src_node = Node(line_arr[0])
            dest_node = Node(line_arr[1])
            if not g.has_node(src_node):
                g.add_node(src_node)
            if not g.has_node(dest_node):
                g.add_node(dest_node)
            edge = WeightedEdge(src_node, dest_node, int(line_arr[2]), int(line_arr[3]))
            g.add_edge(edge)
    return g

# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out


def test_load_map():
    g = load_map('./test_load_map.txt')
    assert str(g) == 'a->b (10, 9)\na->c (12, 2)\nb->c (1, 1)'

# test_load_map()


#
# Problem 3: Finding the Shortest Path using Optimized Search Method
#
# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
# The objective is to minimize the total distance traveled from the start node
# to the end node. The constraint is the maximal distance outdoors that is allowed.

# Problem 3b: Implement get_best_path
def get_best_path(digraph, start, end, path, max_dist_outdoors, best_dist,
                  best_path):
    """
    Finds the shortest path between buildings subject to constraints.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        path: list composed of [[list of strings], int, int]
            Represents the current path of nodes being traversed. Contains
            a list of node names, total distance traveled, and total
            distance outdoors.
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path
        best_dist: int
            The smallest distance between the original start and end node
            for the initial problem that you are trying to solve
        best_path: list of strings
            The shortest path found so far between the original start
            and end node.

    Returns:
        A tuple with the shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k and the distance of that path.

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then return None.
    """
    start_node = Node(start)
    end_node = Node(end)

    if not digraph.has_node(start_node) or not digraph.has_node(end_node):
        raise ValueError('Provided nodes not in the graph')

    if path is None:
        # We're at the beginning.
        path = [[], 0, 0]

    current_path, total_distance_traveled, total_distance_outdoors = path
    current_path = current_path[:]
    current_path += [str(start)]

    if start_node == end_node:
        # We have reached the end, return the shortest path found.
        return [current_path, total_distance_traveled, total_distance_outdoors]

    try:
        edges = digraph.get_edges_for_node(start_node)
    except KeyError:
        edges = []

    for edge in edges:
        dest = edge.get_destination()
        if str(dest) in current_path:
            # Already visited, skip
            continue

        outdoor_distance = edge.get_outdoor_distance()
        remaining_distance_outdoors = max_dist_outdoors - outdoor_distance
        if remaining_distance_outdoors < 0:
            continue

        current_best_dist = total_distance_traveled + edge.get_total_distance()
        if best_dist is None or current_best_dist < best_dist:
            best_dist = current_best_dist

        new_path = get_best_path(
            digraph=digraph,
            start=dest,
            end=end_node,
            path=[
                current_path,
                current_best_dist,
                total_distance_outdoors + outdoor_distance
            ],
            max_dist_outdoors=remaining_distance_outdoors,
            best_dist=current_best_dist,
            best_path=best_path
        )

        if new_path is None:
            continue

        if best_path is None or new_path[1] < best_path[1]:
            best_path = new_path

    return best_path


# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_outdoors):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent outdoors on this path must
    not exceed max_dist_outdoors.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then raises a ValueError.
    """
    shortest_path = get_best_path(
        digraph=digraph,
        start=start,
        end=end,
        path=None,
        best_dist=None,
        best_path=None,
        max_dist_outdoors=max_dist_outdoors,
    )
    if shortest_path is None or shortest_path[1] > max_total_dist:
        raise ValueError('No suitable path could be found')
    return shortest_path[0]


# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class Ps2Test(unittest.TestCase):
    LARGE_DIST = 99999

    def setUp(self):
        self.graph = load_map("mit_map.txt")

    def test_load_map_basic(self):
        self.assertTrue(isinstance(self.graph, Digraph))
        self.assertEqual(len(self.graph.nodes), 37)
        all_edges = []
        for _, edges in self.graph.edges.items():
            all_edges += edges  # edges must be dict of node -> list of edges
        all_edges = set(all_edges)
        self.assertEqual(len(all_edges), 129)

    def _print_path_description(self, start, end, total_dist, outdoor_dist):
        constraint = ""
        if outdoor_dist != Ps2Test.LARGE_DIST:
            constraint = "without walking more than {}m outdoors".format(
                outdoor_dist)
        if total_dist != Ps2Test.LARGE_DIST:
            if constraint:
                constraint += ' or {}m total'.format(total_dist)
            else:
                constraint = "without walking more than {}m total".format(
                    total_dist)

        print("------------------------")
        print("Shortest path from Building {} to {} {}".format(
            start, end, constraint))

    def _test_path(self,
                   expectedPath,
                   total_dist=LARGE_DIST,
                   outdoor_dist=LARGE_DIST):
        start, end = expectedPath[0], expectedPath[-1]
        self._print_path_description(start, end, total_dist, outdoor_dist)
        dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
        print("Expected: ", expectedPath)
        print("DFS: ", dfsPath)
        self.assertEqual(expectedPath, dfsPath)

    def _test_impossible_path(self,
                              start,
                              end,
                              total_dist=LARGE_DIST,
                              outdoor_dist=LARGE_DIST):
        self._print_path_description(start, end, total_dist, outdoor_dist)
        with self.assertRaises(ValueError):
            directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

    def test_path_one_step(self):
        self._test_path(expectedPath=['32', '56'])

    def test_path_no_outdoors(self):
        self._test_path(
            expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

    def test_path_multi_step(self):
        self._test_path(expectedPath=['2', '3', '7', '9'])

    def test_path_multi_step_no_outdoors(self):
        self._test_path(
            expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

    def test_path_multi_step2(self):
        self._test_path(expectedPath=['1', '4', '12', '32'])

    def test_path_multi_step_no_outdoors2(self):
        self._test_path(
            expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
            outdoor_dist=0)

    def test_impossible_path1(self):
        self._test_impossible_path('8', '50', outdoor_dist=0)

    def test_impossible_path2(self):
        self._test_impossible_path('10', '32', total_dist=100)


class TestGetBestPath(unittest.TestCase):
    def test_start_end_same(self):
        g = load_map('./small_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='a',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a'], 0, 0], result)

    def test_start_end_adjacent_nodes(self):
        g = load_map('./small_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='b',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'b'], 2, 1], result)

    def test_start_end_adjacent_nodes2(self):
        g = load_map('./small_graph2.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='b',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'c', 'b'], 4, 2], result)

    def test_start_end_non_adjacent_nodes(self):
        g = load_map('./medium_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='c',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'd', 'e', 'f', 'c'], 6, 4], result)

    def test_start_end_non_adjacent_nodes2(self):
        g = load_map('./medium_graph2.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='c',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'b', 'c'], 9, 2], result)

    def test_start_end_non_adjacent_nodes3(self):
        g = load_map('./medium_graph2.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='e',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'd', 'e'], 5, 4], result)

    def test_start_end_non_adjacent_nodes4(self):
        g = load_map('./medium_graph2.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='f',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'd', 'e', 'f'], 7, 6], result)

    def test_no_path(self):
        g = load_map('./no_path_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='d',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertIsNone(result)

    def test_non_existent_edge(self):
        g = load_map('./no_path_graph.txt')
        with self.assertRaises(ValueError):
            get_best_path(
                digraph=g,
                start='a',
                end='x',
                path=None,
                max_dist_outdoors=20,
                best_dist=None,
                best_path=None
            )

    def test_loop(self):
        g = load_map('./loop_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='e',
            path=None,
            max_dist_outdoors=20,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'c', 'd', 'e'], 3, 3], result)

    def test_max_outdoors(self):
        g = load_map('./max_outdoor_exceeded_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='c',
            path=None,
            max_dist_outdoors=3,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'c'], 5, 2], result)

    def test_max_outdoors2(self):
        g = load_map('./max_outdoor_exceeded_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='c',
            path=None,
            max_dist_outdoors=5,
            best_dist=None,
            best_path=None
        )
        self.assertEqual([['a', 'b', 'c'], 4, 4], result)

    def test_max_outdoors_no_path(self):
        g = load_map('./max_outdoor_exceeded_graph.txt')
        result = get_best_path(
            digraph=g,
            start='a',
            end='c',
            path=None,
            max_dist_outdoors=0,
            best_dist=None,
            best_path=None
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
