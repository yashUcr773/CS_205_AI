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
# for creating random states
import numpy as np
# to track the time used for execution
import time

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


list_of_8_puzzles = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [1, 0, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 0, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 0, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 0, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 0, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 0, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 0, 8],
    [1, 2, 3, 4, 5, 6, 7, 8, 0],
]

list_of_15_puzzles = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0],
]

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
    global_states_manager = []

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
            Node.global_states_manager = []
            Node.global_states_manager.append(
                self._get_state_string(self.state))

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

            if not self._is_state_already_generated(state_copy):
                child_node = Node(self.depth+1, path, state_copy, self)
                Node.global_states_manager.append(
                    self._get_state_string(child_node.state))
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
        includes blank for calculation
        '''

        total_manhattan_distance = 0

        for i in goal_state:
            # goal_state_row, goal_state_colums=self._get_row_col_position(goal_state,i)
            # random_state_row, random_state_colums=self._get_row_col_position(self.state,i)
            # total_manhattan_distance+=abs(goal_state_colums-random_state_colums)+abs(goal_state_row-random_state_row)

            distance = abs(self.state.index(i) - goal_state.index(i))
            total_manhattan_distance += distance/self.row_length + distance%self.row_length

        return int(total_manhattan_distance)

    def misplaced_tile_heuristic(self, goal_state):
        '''
        get value of misplaced tile heuristic for a current state and goal state.
        misplaced tile distance is the count of all the tiles that are not in correct position
        includes blank for calculation
        '''

        misplaced_count = 0
        for i in range(len(goal_state)):
            if self.state[i] != goal_state[i]:
                misplaced_count += 1

        return misplaced_count
    
    def get_heuristic_cost(self, heuristic_measure, goal_state):
        '''
        get the heuristic value cost of expanding this node
        '''

        g_n = self.depth
        h_n = 0

        if heuristic_measure == 'MANHATTAN':
            h_n = self.manhattan_distance_heuristic(goal_state)
        elif heuristic_measure == 'MISPLACED':
            h_n = self.misplaced_tile_heuristic(goal_state)
        elif heuristic_measure == 'UNIFORM':
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
        return state_string in Node.global_states_manager

    def _get_state_string(self, state):
        '''
        convert the child node to a string for unique and easy representation
        '''
        state_string = "".join([str(i) for i in state])
        return state_string

# #############################################################
# ########        BASIC TESTING AND VALIDATION         ########
# #############################################################

# puzz_length = 9
# random_state = generate_random_states(puzz_length)
# random_state = [1,3,6,5,0,2,4,7,8]

# goal_state = [i for i in range(1,puzz_length)]
# goal_state.append(0)

# print ('random_state', random_state)
# print ('goal_state', goal_state)

# parent_node = Node(0, [], random_state, None)

# parent_node.print_state()
# print ('valid moves', parent_node.get_valid_moves())
# print ('manhattan distance', parent_node.manhattan_distance_heuristic(goal_state))
# print ('heuristic distance', parent_node.misplaced_tile_heuristic(goal_state))

# print ('\n... solver ...\n')
# def spawn_node_children(parent, goal, limit):

#     if limit == 0:
#         return

#     # print (parent.state)
#     # print (goal)
#     if parent.state == goal:
#         print ('----------- goal found ---------')
#         parent.print_state(True)
#         return

#     child = parent.spawn_children()
#     for c in child:
#         # c.print_state(True)
#         spawn_node_children(c, goal, limit-1)

# parent_node.print_state(True)
# spawn_node_children(parent_node,goal_state, 15)

#############################################################
########               GENERAL SEARCH                ########
#############################################################


def make_queue(node: Node, goal_state:list, heuristic_measure: str):
    '''
    Initialize an empty queue.
    take in a node and add it to the queue.
    return the queue
    '''
    return [(node.get_heuristic_cost(heuristic_measure, goal_state),node)]


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
        child_queue.append((child.get_heuristic_cost(heuristic_measure, goal_state),child))
    
    return queue + child_queue


def general_search(initial_state, goal_state, queueing_function, heuristic_measure):
    '''
    # general search function
    # refered from the problem statement doc provided.
    # link to doc
    # https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_1_The_Eight_Puzzle_CS_205.pdf
    # takes in node problem and a queueing function and solves the problem using the queueing function
    # problem has initial_state, final_state and supported operators
    # queueing function adds the node in queue as required
    '''

    nodes = make_queue(make_node_from_state(initial_state), goal_state, heuristic_measure)

    while True:
        if is_queue_empty(nodes):
            return 'FAILURE'
        else:
            nodes, node = remove_front(nodes)

            if goal_state == node.state:
                node.print_state(True)
                return 'SUCCESS'
            else:
                nodes = queueing_function(nodes, expand_nodes(node), heuristic_measure, goal_state)

trial_puzzles = [
    [1,2,3,4,5,6,7,8,0],
    [1,2,3,4,5,6,0,7,8],
    [1,2,3,5,0,6,4,7,8],
    [1,3,6,5,0,2,4,7,8],
    [1,3,6,5,0,7,4,8,2],
    [1,6,7,5,0,3,4,8,2],
    [7,1,2,4,8,5,6,3,0],
    [0,7,2,4,6,1,3,5,8],
]

# #############################################################
# ########       SEARCH TESTING AND VALIDATION         ########
# #############################################################

for i in trial_puzzles:
    puzz_length = 9
    random_state = generate_random_states(puzz_length)
    # random_state = [1,3,6,5,0,2,4,7,8]
    random_state = i

    goal_state = [i for i in range(1,puzz_length)]
    goal_state.append(0)

    print ('random_state', random_state)
    print ('goal_state', goal_state)

    t0 = time.time()
    general_search(random_state, goal_state, queueing_function, 'MANHATTAN')
    t1 = time.time()
    
    print ('time', t1 - t0)
    print ('--------------')




# #############################################################
# ########      LANDING PAGE AND INPUT VALIDATION      ########
# #############################################################

# def main_block(clear_previous=True):
#     '''
#     print out landing page.
#     get algo choice
#     get input state
#     validate input
#     call the search function
#     '''

#     puzzle_state = 8

#     # clear screen before landing page
#     if clear_previous:
#         os.system('cls')

#     # main block
#     # Get algo choice
#     print(f'---- {puzzle_state} puzzle solver ----')
#     print('1. Uniform Cost Search')
#     print('2. A* with Misplaced Tile')
#     print('3. A* with Manahattan Distance')
#     algo_choice = int(input('Enter choice: '))

#     if algo_choice not in [1, 2, 3]:
#         os.system('cls')
#         print('Please enter correct choice.\n')
#         main_block(clear_previous=False)
#         return

#     # get input puzzle state
#     # TODO : Add code to choose between default puzzle and input state
#     # TODO : Add code to let user input goal state as well
#     print('\nEnter the numbers in puzzle as space seperated list.')
#     print('Represent blank with 0')
#     print('For Example: 1 2 3 4 0 5 6 7 8\n')
#     problem_input = input('Numbers: ')
#     problem_state = problem_input.split(' ')

#     # convert string to integers
#     problem_state = [int(i) for i in problem_state]

#     if not validate_state(problem_state):
#         os.system('cls')
#         print('Pleae enter valid input state.\n')
#         main_block(clear_previous=False)
#         return

#     os.system('cls')
#     print('Initial State\n')
#     parent_node = Node(0, [], problem_state, None)
#     parent_node.print_state()

#     if algo_choice == 1:
#         print('\nSolving for Uniform cost\n')
#     elif algo_choice == 1:
#         print('\nSolving for A* with Misplaced Tile\n')
#     elif algo_choice == 1:
#         print('\nSolving for A* with Manahattan Distance\n')

#     return


# main_block()