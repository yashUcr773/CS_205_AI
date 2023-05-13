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
            goal_state_row, goal_state_colums = self._get_row_col_position(
                goal_state, i)
            random_state_row, random_state_colums = self._get_row_col_position(
                self.state, i)
            total_manhattan_distance += abs(goal_state_colums - random_state_colums) + abs(
                goal_state_row - random_state_row)

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

#############################################################
########               GENERAL SEARCH                ########
#############################################################


def make_queue(node: Node):
    '''
    Initialize an empty queue.
    take in a node and add it to the queue.
    return the queue
    '''
    return []


def is_queue_empty(queue: list):
    '''
    take in queue and check if the queue is empty
    '''
    pass


def remove_front(queue: list):
    '''
    take in queue and remove the first node
    return the first removed node
    '''
    node = Node(1, 2, 3, 4)
    return node


def expand_nodes(node: Node, operators: list):
    '''
    takes in node and operators and expands the node based on operators.
    returns a list of nodes
    '''
    pass


def make_node_from_state(state: list):
    '''
    call in the Node function to spawn children after making vlaid moves
    '''
    node = Node(1, 2, 3, 4)
    return node


def general_search(problem, queueing_function):
    '''
    # general search function
    # refered from the problem statement doc provided.
    # link to doc
    # https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_1_The_Eight_Puzzle_CS_205.pdf
    # takes in node problem and a queueing function and solves the problem using the queueing function
    # problem has initial_state, final_state and supported operators
    # queueing function adds the node in queue as required
    '''

    nodes = make_queue(make_node_from_state(problem.initial_state))

    while True:
        if is_queue_empty(nodes):
            return 'FAILURE'
        else:
            node = remove_front(nodes)

            if problem.final_state(node.state):
                return 'SUCCESS'
            else:
                nodes = queueing_function(
                    nodes, expand_nodes(node, problem.operators))


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


#############################################################
########      LANDING PAGE AND INPUT VALIDATION      ########
#############################################################

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


def main_block(clear_previous=True):
    '''
    print out landing page.
    get algo choice
    get input state
    validate input
    call the search function
    '''

    puzzle_state = 8

    # clear screen before landing page
    if clear_previous:
        os.system('cls')

    # main block
    # Get algo choice
    print(f'---- {puzzle_state} puzzle solver ----')
    print('1. Uniform Cost Search')
    print('2. A* with Misplaced Tile')
    print('3. A* with Manahattan Distance')
    algo_choice = int(input('Enter choice: '))

    if algo_choice not in [1, 2, 3]:
        os.system('cls')
        print('Please enter correct choice.\n')
        main_block(clear_previous=False)
        return

    # get input puzzle state
    # TODO : Add code to choose between default puzzle and input state
    # TODO : Add code to let user input goal state as well
    print('\nEnter the numbers in puzzle as space seperated list.')
    print('Represent blank with 0')
    print('For Example: 1 2 3 4 0 5 6 7 8\n')
    problem_input = input('Numbers: ')
    problem_state = problem_input.split(' ')

    # convert string to integers
    problem_state = [int(i) for i in problem_state]

    if not validate_state(problem_state):
        os.system('cls')
        print('Pleae enter valid input state.\n')
        main_block(clear_previous=False)
        return

    os.system('cls')
    print('Initial State\n')
    parent_node = Node(0, [], problem_state, None)
    parent_node.print_state()

    if algo_choice == 1:
        print('\nSolving for Uniform cost\n')
    elif algo_choice == 1:
        print('\nSolving for A* with Misplaced Tile\n')
    elif algo_choice == 1:
        print('\nSolving for A* with Manahattan Distance\n')

    return


main_block()
