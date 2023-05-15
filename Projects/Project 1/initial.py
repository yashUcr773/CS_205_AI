'''
Solve 8 puzzle with
Uniform cost search
A* with misplaced tile
A* with manhattan distance
'''

#############################################################
########               LIBRARY IMPORTS               ########
#############################################################

# for square root of numbers
import math
# clear output screen
import os
# to make deep copies of nodes' states
import copy as cpy
# to track the time used for execution
import time
# for creating random states
import numpy as np
# for plotting results and graphs
import matplotlib.pyplot as plt

#############################################################
########                  CONSTANTS                  ########
#############################################################
UNIFORM = 'Uniform'
MISPLACED = 'Misplaced'
MANHATTAN = 'Manhattan'

color_map = {
    MANHATTAN: 'red',
    MISPLACED: 'green',
    UNIFORM: 'blue',
}

direction_map = {
    'R': 'Right',
    'L': 'Left',
    'U': 'Up',
    'D': 'Down'
}

#############################################################
########               LIST OF PUZZLES               ########
#############################################################

list_of_easy_puzzles = [
    ([1, 2, 3, 4, 5, 6, 7, 8, 0], 0),
    ([1, 2, 3, 4, 5, 6, 7, 0, 8], 1),
    ([1, 2, 3, 4, 5, 6, 0, 7, 8], 2),
    ([1, 0, 3, 4, 2, 5, 7, 8, 6], 3),
    ([1, 2, 3, 5, 0, 6, 4, 7, 8], 4),
    ([2, 0, 3, 1, 5, 6, 4, 7, 8], 5),
    ([2, 5, 3, 1, 0, 6, 4, 7, 8], 6),
    ([4, 1, 2, 5, 8, 3, 7, 0, 6], 7),
    ([1, 3, 6, 5, 0, 2, 4, 7, 8], 8),
    ([2, 5, 3, 0, 7, 6, 1, 4, 8], 9),
]

list_of_medium_puzzles = [
    ([2, 3, 5, 1, 4, 6, 0, 7, 8], 10),
    ([1, 4, 2, 7, 8, 3, 5, 0, 6], 11),
    ([1, 3, 6, 5, 0, 7, 4, 8, 2], 12),
    ([7, 2, 3, 0, 5, 6, 1, 4, 8], 13),
    ([1, 5, 0, 3, 2, 4, 7, 8, 6], 14),
    ([4, 2, 1, 0, 3, 6, 7, 5, 8], 15),
    ([1, 6, 7, 5, 0, 3, 4, 8, 2], 16),
    ([4, 8, 1, 0, 3, 5, 2, 7, 6], 17),
    ([7, 5, 3, 1, 4, 6, 2, 8, 0], 18),
    ([5, 4, 6, 3, 1, 2, 7, 0, 8], 19),
]

list_of_hard_puzzles = [
    ([7, 1, 2, 4, 8, 5, 6, 3, 0], 20),
    ([3, 5, 4, 8, 7, 0, 2, 6, 1], 21),
    ([7, 1, 8, 5, 0, 3, 6, 4, 2], 22),
    ([1, 0, 8, 4, 7, 2, 3, 5, 6], 23),
    ([0, 7, 2, 4, 6, 1, 3, 5, 8], 24),
    ([5, 2, 1, 0, 8, 4, 7, 3, 6], 25),
    ([6, 3, 1, 4, 0, 7, 8, 2, 5], 26),
    ([4, 0, 7, 2, 6, 5, 8, 1, 3], 27),
    ([8, 6, 4, 2, 0, 7, 3, 1, 5], 28),
    ([5, 2, 1, 3, 8, 4, 6, 0, 7], 29),
    ([6, 4, 7, 8, 3, 5, 1, 2, 0], 30),
]

#############################################################
########              Utility Functions              ########
#############################################################


def generate_random_states(state_len):
    '''
    generate random states for evaulating and testing
    '''
    arr = [i for i in range(state_len)]
    np.random.shuffle(arr)
    return arr


def validate_state(problem_state: list) -> bool:
    '''
    check if the state passed is valid or not.
    checks that state length is a perfect sqaure.
    checks that all numbers are unique.
    checks that numbers are in range 0-len(state)
    '''

    # get length of puzzle
    # is it 8 puzzle, 15 puzzle etc
    puzzle_size = len(problem_state)

    # check that length is perfect square
    sqrt = int(math.sqrt(puzzle_size))
    if(sqrt*sqrt) != puzzle_size:
        return False

    # generate valid inputs.
    # valid inputs range from 0 - n for n puzzle
    valid_inputs = [i for i in range(puzzle_size)]

    # put the problem state in a set to remove duplicates.
    problem_state_set = set(problem_state)

    for i in problem_state_set:
        if i not in valid_inputs:
            return False

    return True


def print_formatted_time(time_input):
    hrs = int(time_input // 3600)
    mins = int((time_input % 3600) // 60)
    secs = int((time_input % 3600) % 60)
    if hrs:
        print(f'time taken is {hrs} hrs, {mins} mins and {secs} secs')
    elif mins:
        print(f'time taken is {mins} mins and {secs} secs')
    else:
        print(f'time taken is {secs} secs')


def print_time(time_input):
    if time_input <= 1e-5:
        print(f'time taken is {time_input:.6f} secs')
    elif time_input <= 1e-4:
        print(f'time taken is {time_input:.5f} secs')
    elif time_input <= 1e-3:
        print(f'time taken is {time_input:.4f} secs')
    elif time_input <= 1e-2:
        print(f'time taken is {time_input:.3f} secs')
    elif time_input <= 1e-1:
        print(f'time taken is {time_input:.2f} secs')
    elif time_input >= 0 and time_input <= 1:
        print(f'time taken is {time_input} secs')
    else:
        print_formatted_time(time_input)


def print_trace(node, goal_state):
    if node == None:
        return

    print_trace(node.parent, goal_state)
    node.print_trace_info(goal_state)

#############################################################
########    NODE CLASS AND CORRESPONDING FUNCTION    ########
#############################################################


class Node():

    '''
    create a node class for the states.
    stores path to current node from parent, depth etc and other properties.
    has uitlity methods.
    '''

    # to store the states that have already been generated.
    # prevents exploring repeating states.
    # If a state is present here, then it has already been generated at higher depth
    global_states_manager = {}

    # initialize the node
    # depth of node.
    # path stores the path to be taken to reach till current node.
    # state of node
    # link to parent node
    def __init__(self, depth, path, state, parent):
        self.depth = depth
        self.path = path
        self.state = state
        self.parent = parent

        # puzzle length 8/15/24
        self.state_length = len(state)
        # length per row
        self.row_length = int(math.sqrt(self.state_length))
        # length per column
        self.col_length = int(math.sqrt(self.state_length))

        # in case the current node is parent node, empty the globally stored states.
        if self.parent is None:
            Node.global_states_manager = {}
            Node.global_states_manager[self._get_state_string(
                self.state)] = self.depth

    def spawn_children(self):
        '''
        create child nodes after making valid moves on parent node.
        children are not generated is their states are already present in global states manager
        '''

        # get index of blank tile
        blank_idx = self.state.index(0)

        children_list = []

        for move in self.get_valid_moves():
            state_copy = cpy.deepcopy(self.state)
            path = cpy.deepcopy(self.path)

            if move == 'U':
                path.append('U')
                state_copy[blank_idx], state_copy[blank_idx -
                                                  self.row_length] = state_copy[blank_idx-self.row_length], state_copy[blank_idx]

            elif move == 'L':
                path.append('L')
                state_copy[blank_idx], state_copy[blank_idx -
                                                  1] = state_copy[blank_idx-1], state_copy[blank_idx]

            elif move == 'R':
                path.append('R')
                state_copy[blank_idx], state_copy[blank_idx +
                                                  1] = state_copy[blank_idx+1], state_copy[blank_idx]

            elif move == 'D':
                path.append('D')
                state_copy[blank_idx], state_copy[blank_idx +
                                                  self.row_length] = state_copy[blank_idx+self.row_length], state_copy[blank_idx],

            past_depth_if_generated = self._is_state_already_generated(
                state_copy)
            if past_depth_if_generated == -1 or past_depth_if_generated > self.depth+1:
                child_node = Node(self.depth+1, path, state_copy, self)
                Node.global_states_manager[self._get_state_string(
                    child_node.state)] = self.depth+1
                children_list.append(child_node)

        return children_list

    def get_valid_moves(self):
        '''
        get list of valid operators for each puzzle state
        '''

        # total Valid moves.
        # Move the blank space in following directions
        # Up, Left, Right, Down
        valid = ['U', 'L', 'R', 'D']

        # get index of blank tile
        blank_idx = self.state.index(0)

        # if the blank tile is in first row, cant move up
        if blank_idx >= 0 and blank_idx < self.col_length:
            valid.remove('U')

        # if the blank tile is in last row, cant move down
        if blank_idx >= (self.col_length*self.col_length - self.col_length) and blank_idx < self.col_length*self.col_length:
            valid.remove('D')

        # if the blank tile is in first column, cant move left
        if blank_idx % self.col_length == 0:
            valid.remove('L')

        # if the blank tile is in last column, cant move right
        if (blank_idx + 1) % self.col_length == 0:
            valid.remove('R')

        return valid

    def manhattan_distance_heuristic(self, goal_state):
        '''
        get value for manhattan distance for a current state and goal state
        manhattan distance is the shortest distance a tile needs to be moved to get to correct position
        total distance is sum of all individual distances
        does not include blank for calculation
        '''

        total_manhattan_distance = 0

        for i in goal_state:
            if i == 0:
                continue

            goal_state_row, goal_state_colums = self._get_row_col_position(
                goal_state, i)
            random_state_row, random_state_colums = self._get_row_col_position(
                self.state, i)
            total_manhattan_distance += abs(goal_state_colums-random_state_colums)+abs(
                goal_state_row-random_state_row)

        return int(total_manhattan_distance)

    def misplaced_tile_heuristic(self, goal_state):
        '''
        get value of misplaced tile heuristic for a current state and goal state.
        misplaced tile distance is the count of all the tiles that are not in correct position
        does not include blank for calculation
        '''

        misplaced_count = 0
        for i in range(len(goal_state)):
            if i == 0:
                continue

            if self.state[i] != goal_state[i]:
                misplaced_count += 1

        return misplaced_count

    def get_heuristic_cost(self, heuristic_measure, goal_state):
        '''
        get the heuristic value cost of expanding this node
        '''

        g_n = self.depth
        h_n = 0

        if heuristic_measure == MANHATTAN:
            h_n = self.manhattan_distance_heuristic(goal_state)
        elif heuristic_measure == MISPLACED:
            h_n = self.misplaced_tile_heuristic(goal_state)
        elif heuristic_measure == UNIFORM:
            h_n = 0
        else:
            h_n = 0

        return g_n + h_n

    def print_state(self, verbose=False):
        '''
        # takes in list of numbers and print it in puzzle view
        '''

        if verbose:
            print('depth', self.depth)
            print('path', self.path)

        self._print_horizontal_divider(self.state_length)
        for i in range(self.state_length):
            print(f'| {self.state[i]:2} |', end="")
            if (i+1) % self.row_length == 0:
                self._print_horizontal_divider(self.state_length)
        print()

    def print_trace_info(self, goal_state):

        if len(self.path):
            print(
                f'The best state to expand with g(n): {self.depth} and h(n): {self.manhattan_distance_heuristic(goal_state)}')
            print('Move blank to: ', direction_map[self.path[-1]])
            print('Updated State', end='')
        else:
            print('\nProblem State', end='')

        self._print_horizontal_divider(self.state_length)
        for i in range(self.state_length):
            print(f'| {self.state[i]:2} |', end="")
            if (i+1) % self.row_length == 0:
                self._print_horizontal_divider(self.state_length)
        print()

    def _get_row_col_position(self, state, element):
        '''
        get row and column positions for a given element in a given state
        used for manhattan distance
        '''

        idx = state.index(element)
        column_val = int(idx % self.row_length)
        r_val = int(idx // self.row_length)
        return r_val, column_val

    def _print_horizontal_divider(self, size=8):
        '''
        print horizontal dividers after each row for better UI
        '''
        if size == 9:
            print('\n------------------')
        elif size == 16:
            print('\n------------------------')

    def _is_state_already_generated(self, state):
        '''
        check if the potential child state is already generated
        '''
        state_string = "".join([str(i) for i in state])
        return Node.global_states_manager[state_string] if state_string in Node.global_states_manager else -1

    def _get_state_string(self, state):
        '''
        convert the child node to a string for unique and easy representation
        '''
        state_string = "".join([str(i) for i in state])
        return state_string

#############################################################
########               GENERAL SEARCH                ########
#############################################################


def make_queue(node: Node, goal_state: list, heuristic_measure: str):
    '''
    Initialize an empty queue.
    take in a node and add it to the queue.
    return the queue
    '''
    return [(node.get_heuristic_cost(heuristic_measure, goal_state), node)]


def is_queue_empty(queue: list):
    '''
    take in queue and check if the queue is empty
    '''
    return False if len(queue) > 0 else True


def remove_front(queue: list):
    '''
    take in queue and remove the first node
    return the first removed node
    '''

    queue = sorted(queue, key=lambda x: x[0])
    node = queue.pop(0)[1]
    return queue, node


def expand_nodes(node: Node):
    '''
    takes in node and operators and expands the node based on operators.
    returns a list of nodes
    '''

    children = node.spawn_children()
    return children


def make_node_from_state(state: list):
    '''
    call in the Node class to create Parent Node
    '''
    parent_node = Node(0, [], state, None)
    return parent_node


def queueing_function(queue, children, heuristic_measure, goal_state):
    '''
    take in queue,
    take in children
    put children in priority queue based on heuristic value
    return the queue
    '''

    child_queue = []
    for child in children:
        child_queue.append((child.get_heuristic_cost(
            heuristic_measure, goal_state), child))

    queue = queue + child_queue
    queue = sorted(queue, key=lambda x: x[0])

    return queue


def general_search(initial_state, goal_state, queueing_function, heuristic_measure, verbose=False):
    '''
    # general search function
    # refered from the problem statement doc provided.
    # link to doc
    # https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_1_The_Eight_Puzzle_CS_205.pdf
    # takes in node problem and a queueing function and solves the problem using the queueing function
    # problem has initial_state, final_state and supported operators
    # queueing function adds the node in queue as required
    '''

    nodes = make_queue(make_node_from_state(initial_state),
                       goal_state, heuristic_measure)

    total_nodes_expanded = 0
    max_queue_size = 0

    while True:

        max_queue_size = max(max_queue_size, len(nodes))

        if is_queue_empty(nodes):
            print("FAILURE")
            return -1, total_nodes_expanded, max_queue_size
        else:
            nodes, node = remove_front(nodes)

            if goal_state == node.state:
                if verbose:
                    print("SUCCESS")
                    node.print_state(True)
                    print('Total nodes Expanded : ', total_nodes_expanded)
                    print('Max Queue Size : ', max_queue_size)
                return node, max_queue_size, total_nodes_expanded
            else:
                total_nodes_expanded += 1
                nodes = queueing_function(nodes, expand_nodes(
                    node), heuristic_measure, goal_state)

# #############################################################
# ####    STAND_ALONE SEARCH TESTING AND VALIDATION    ########
# #############################################################

# for puzzle, true_depth in list_of_easy_puzzles:
#     goal_state = [i for i in range(1, len(puzzle))]
#     goal_state.append(0)

#     print('true_depth', true_depth)
#     print('random_state', puzzle)
#     print('goal_state', goal_state)

#     t0 = time.time()
#     general_search(puzzle, goal_state, queueing_function, MANHATTAN, verbose=True)
#     t1 = time.time()

#     print('time', t1 - t0)

# #############################################################
# ####          Generate and display Traceback         ########
# #############################################################

# puzzle, true_depth = list_of_easy_puzzles[np.random.choice(len(list_of_easy_puzzles))]

# goal_state = [i for i in range(1, len(puzzle))]
# goal_state.append(0)

# print(f'The True Depth of the puzzle is: {true_depth}\n')

# final_node, _, _ = general_search(puzzle, goal_state, queueing_function, MANHATTAN, verbose=False)

# print_trace(final_node, goal_state)

# # #############################################################
# # ########  UI LANDING PAGE AND INPUT VALIDATION       ########
# # #############################################################


# def main_block(clear_previous=True):
#     '''
#     print out landing page.
#     get algo choice
#     get input state
#     validate input
#     call the search function
#     '''

#     # clear screen before landing page
#     if clear_previous:
#         os.system('cls')

#     # main block
#     # Get algo choice
#     print('---- N Puzzle Solver ----')
#     print('1. Uniform Cost Search')
#     print('2. A* with Misplaced Tile')
#     print('3. A* with Manahattan Distance')
#     algo_choice = int(input('Enter choice: '))

#     if algo_choice not in [1, 2, 3]:
#         os.system('cls')
#         print('Please enter correct choice.\n')
#         main_block(clear_previous=False)
#         return

#     # Get Puzzle choice
#     print('\n---- Choose Puzzle Type ----')
#     print('1. Random Default Easy (depth 0-9)')
#     print('2. Random Default Medium (depth 10-19)')
#     print('3. Random Default Hard (depth >20)')
#     print('4. Custom Puzzle')
#     puzzle_choice = int(input('Enter choice: '))

#     if puzzle_choice not in [1, 2, 3, 4]:
#         os.system('cls')
#         print('Please enter correct choice.\n')
#         main_block(clear_previous=False)
#         return

#     # get input puzzle state
#     problem_state = []
#     if puzzle_choice == 1:
#         problem_state = list_of_easy_puzzles[np.random.choice(len(list_of_easy_puzzles))][0]
#     elif puzzle_choice == 2:
#         problem_state = list_of_medium_puzzles[np.random.choice(len(list_of_medium_puzzles))][0]
#     elif puzzle_choice == 3:
#         problem_state = list_of_hard_puzzles[np.random.choice(len(list_of_hard_puzzles))][0]
#     elif puzzle_choice == 4:
#         print('\nEnter the numbers in puzzle as space seperated list.')
#         print('Represent blank with 0')
#         print('For Example: 1 2 3 4 0 5 6 7 8\n')
#         problem_input = input('Numbers: ')
#         problem_state = problem_input.split(' ')

#         # convert string to integers
#         problem_state = [int(i) for i in problem_state]

#         if not validate_state(problem_state):
#             os.system('cls')
#             print('Pleae enter valid input state.\n')
#             main_block(clear_previous=False)
#             return

#     # Get Puzzle choice
#     print('\n---- Enter Goal State ----')
#     print('1. Default State (1 2 3 4 5 6 ... n-1 n 0)')
#     print('2. Custom Goal')
#     puzzle_choice = int(input('Enter choice: '))

#     if puzzle_choice not in [1, 2]:
#         os.system('cls')
#         print('Please enter correct choice.\n')
#         main_block(clear_previous=False)
#         return

#     # get goal state
#     goal_state = []
#     if puzzle_choice == 1:

#         goal_state = list(range(1, len(problem_state)))
#         goal_state.append(0)

#     elif puzzle_choice == 4:
#         print('\nEnter the numbers in puzzle as space seperated list.')
#         print('Represent blank with 0')
#         print('For Example: 1 2 3 4 0 5 6 7 8\n')
#         problem_input = input('Numbers: ')
#         goal_state = problem_input.split(' ')

#         # convert string to integers
#         goal_state = [int(i) for i in goal_state]

#         if not validate_state(goal_state):
#             os.system('cls')
#             print('Pleae enter valid input state.\n')
#             main_block(clear_previous=False)
#             return

#     os.system('cls')
#     print('Initial State\n')
#     parent_node = Node(0, [], problem_state, None)
#     parent_node.print_state()

#     if algo_choice == 1:

#         print('\nSolving for Uniform cost\n')
#         time_before = time.time()
#         final_node, _, _ = general_search(problem_state, goal_state, queueing_function, UNIFORM, verbose=True)
#         time_after = time.time()
#         total_time = time_after - time_before
#         print_time(total_time)

#     elif algo_choice == 2:

#         print('\nSolving for A* with Misplaced Tile\n')
#         time_before = time.time()
#         final_node, _, _ = general_search(problem_state, goal_state, queueing_function, MISPLACED, verbose=True)
#         time_after = time.time()
#         total_time = time_after - time_before
#         print_time(total_time)


#     elif algo_choice == 3:

#         print('\nSolving for A* with Manahattan Distance\n')
#         time_before = time.time()
#         final_node, _, _ = general_search(problem_state, goal_state, queueing_function, MANHATTAN, verbose=True)
#         time_after = time.time()
#         total_time = time_after - time_before
#         print_time(total_time)

#     # Print Traceback
#     print('\n---- Want to print the puzzle traceback? ----')
#     print('1. Yes')
#     print('2. No. Exit')
#     traceback_choice = int(input('Enter choice: '))

#     if traceback_choice not in [1, 2]:
#         os.system('cls')
#         print('Please enter correct choice.\n')
#         main_block(clear_previous=False)
#         return

#     if traceback_choice == 1:
#         print_trace(final_node, goal_state)


#     return


# main_block()

# #############################################################
# ########     Multiple TESTS and Result Analysis      ########
# #############################################################

# combined_puzzles_list = list_of_easy_puzzles + list_of_medium_puzzles + list_of_hard_puzzles
# goal_state = [1,2,3,4,5,6,7,8,0]

# time_collection = {}
# queue_collection = {}
# nodes_collection = {}

# for heuristic in [MANHATTAN, UNIFORM, MISPLACED]:
#     time_collection[heuristic] = []
#     queue_collection[heuristic] = []
#     nodes_collection[heuristic] = []


# for puzzle, true_depth in combined_puzzles_list:
#     for heuristic in [MANHATTAN]:
#         print (heuristic, puzzle, true_depth)
#         time_before = time.time()
#         final_node, max_queue, total_nodes = general_search(puzzle, goal_state, queueing_function, heuristic, verbose=False)
#         time_after = time.time()
#         total_time = time_after - time_before
#         print (final_node.depth)

#         time_collection[heuristic].append((true_depth, total_time))
#         queue_collection[heuristic].append((true_depth, max_queue))
#         nodes_collection[heuristic].append((true_depth, total_nodes))
#     print ()

# plt.figure(1)
# for heuristic in [MANHATTAN, UNIFORM, MISPLACED]:
#     temp_arr = np.array(time_collection[heuristic])
#     plt.plot(temp_arr[:,0], temp_arr[:,1], color = color_map[heuristic], label=heuristic)

# plt.title('Time vs Depth')
# plt.xlabel('depth')
# plt.ylabel('time in seconds')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure(2)
# for heuristic in [MANHATTAN, UNIFORM, MISPLACED]:
#     temp_arr = np.array(nodes_collection[heuristic])
#     plt.plot(temp_arr[:,0], temp_arr[:,1], color = color_map[heuristic], label=heuristic)

# plt.title('Nodes Expanded vs Depth')
# plt.xlabel('depth')
# plt.ylabel('Nodes Expanded')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure(3)
# for heuristic in [MANHATTAN, UNIFORM, MISPLACED]:
#     temp_arr = np.array(queue_collection[heuristic])
#     plt.plot(temp_arr[:,0], temp_arr[:,1], color = color_map[heuristic], label=heuristic)

# plt.title('Max Queue Size vs Depth')
# plt.xlabel('depth')
# plt.ylabel('Max Queue Size')
# plt.grid()
# plt.legend()
# plt.show()

#############################################################
########    Generating puzzles at different depths   ########
#############################################################

# https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/

def getInvCount(arr):
    inv_count = 0
    empty_value = 0
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if arr[j] != empty_value and arr[i] != empty_value and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


def isSolvable(puzzle):

    # Count inversions in given 8 puzzle
    inv_count = getInvCount(puzzle)

    # return true if inversion count is even.
    return (inv_count % 2 == 0)


puzzle_book = {}
for i in range(30000):

    random_state = generate_random_states(9)
    if (isSolvable(random_state)):

        goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        final_node, _, _ = general_search(
            random_state, goal_state, queueing_function, MANHATTAN, verbose=False)

        # if final_node.depth not in puzzle_book:
        #     puzzle_book[final_node.depth] = []

        # puzzle_book[final_node.depth].append(random_state)
        if final_node.depth == 28:
            print(random_state)

print(puzzle_book)

# #############################################################
# ########  Pretty Print Puzzles to display in Report  ########
# #############################################################

# combined_puzzles = list_of_easy_puzzles + list_of_medium_puzzles + list_of_hard_puzzles
# combined_puzzles = list_of_hard_puzzles
# for puzzle, true_depth in combined_puzzles:
#     print (f'          --- {true_depth} ---')
#     print ('         ', puzzle[:3])
#     print ('         ', puzzle[3:6])
#     print ('         ', puzzle[6:9])
#     print ()
#     print ()
