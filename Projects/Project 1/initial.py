# for square root
import math

# clear output screen
import os 

# for creating random states
import numpy as np

# Solve 8 puzzle with
# Uniform cost search
# A* with misplaced tile
# A* with manhattan distance

# General search

# take in a state and create node from it
# node has following properties
# ...
def make_node_from_state(state):
    pass

# Initialize an empty queue.
# take in a node and add it to the queue.
# return the queue
def make_queue(node):
    pass

# take in queue and check if the queue is empty
def is_queue_empty(queue):
    pass

# take in queue and remove the first node
# return the first removed node
def remove_front(queue):
    pass

# takes in node and operators and expands the node based on operators.
# returns a list of nodes
def expand_nodes(node, operators):
    pass

# general search function
# refered from the problem statement doc provided.
# link to doc -> https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_1_The_Eight_Puzzle_CS_205.pdf
# takes in node problem and a queueing function and solves the problem using the queueing function
# problem has initial_state, final_state and supported operators
# queueing function adds the node in queue as required
def general_search(problem, queueing_function):
    nodes = make_queue(make_node_from_state(problem.initial_state))

    while True:
        if is_queue_empty(nodes):
            return 'FAILURE'
        else:
            node = remove_front(nodes)

            if problem.final_state(node.state):
                return 'SUCCESS'
            else:
                nodes = queueing_function(nodes, expand_nodes(node, problem.operators))

# takes in list of numbers and validates if the input is valid or not
def validate_numbers_input(problem_state):

    # get length of puzzle
    # is it 8 puzzle, 15 puzzle etc
    puzzle_size = len(problem_state)

    # generate valid inputs.
    # valid inputs range from 0 - n for n puzzle
    valid_inputs = [i for i in range(puzzle_size)]

    # put the problem state in a set to remove duplicates.
    problem_state_set = set(problem_state)
    
    for i in problem_state_set:
        if i not in valid_inputs:
            return False
    
    return True

# print horizontal dividers after each row for better UI
def print_horizontal_divider(size = 8):
    if size == 9:
        print ('\n------------------')
    elif size == 16:
        print ('\n------------------------')

# takes in list of numbers and print it in puzzle view
def print_state(problem_state):

    if validate_numbers_input(problem_state):
        
        # get length of puzzle
        # is it 8 puzzle, 15 puzzle etc
        puzzle_size = len(problem_state)

        # get total number of elements per row/column
        sqrt = int(math.sqrt(puzzle_size))

        print_horizontal_divider(puzzle_size)

        if sqrt*sqrt != puzzle_size:
            raise Exception (f'Enter Valid Puzzle Format, {puzzle_size-1} is not valid')
        else:
            for i in range(len(problem_state)):
                print (f'| {problem_state[i]:2} |', end = "")
                if (i+1) % sqrt == 0:
                    print_horizontal_divider(puzzle_size)
        print ()
    else:
        raise Exception ('Enter Valid Puzzle State')

# get list of valid operators for each puzzle state
def getValidOperators(state):

    # total Valid moves. 
    # Move the blank space in following directions
    # Up, Left, Right, Down
    valid = ['U', 'L', 'R', 'D']

    # get length of puzzle
    # is it 8 puzzle, 15 puzzle etc
    puzzle_size = len(state)

    # get total number of elements per row/column
    sqrt = int(math.sqrt(puzzle_size))

    # get index of blank tile
    blank_idx = state.index(0)

    # if the blank tile is in first row, cant move up
    if blank_idx >=0 and blank_idx<sqrt:
        valid.remove('U')
    
    # if the blank tile is in last row, cant move down
    if blank_idx >= (sqrt*sqrt - sqrt) and blank_idx<sqrt*sqrt:
        valid.remove('D')

    # if the blank tile is in first column, cant move left
    if blank_idx % sqrt == 0:
        valid.remove('L')
    
    # if the blank tile is in last column, cant move right
    if (blank_idx + 1) % sqrt == 0:
        valid.remove('R')

    return valid

# get sub puzzles for each puzzle after making a valid move
def make_valid_moves(state, valid_moves):

    # get length of puzzle
    # is it 8 puzzle, 15 puzzle etc
    puzzle_size = len(state)

    # get total number of elements per row/column
    sqrt = int(math.sqrt(puzzle_size))

    # get index of blank tile
    blank_idx = state.index(0)

    for move in valid_moves:
        state_copy = copy.deepcopy(state)

        if move == 'U':
            state_copy[blank_idx], state_copy[blank_idx-sqrt] = state_copy[blank_idx-sqrt], state_copy[blank_idx]
            print ('U')
            print_state(state_copy)
        
        elif move == 'L':
            state_copy[blank_idx], state_copy[blank_idx-1] =  state_copy[blank_idx-1], state_copy[blank_idx]
            print ('L')
            print_state(state_copy)

        elif move == 'R':
            state_copy[blank_idx], state_copy[blank_idx+1] =  state_copy[blank_idx+1], state_copy[blank_idx]
            print ('R')
            print_state(state_copy)

        elif move == 'D':
            state_copy[blank_idx], state_copy[blank_idx+sqrt] =  state_copy[blank_idx+sqrt], state_copy[blank_idx],
            print ('D')
            print_state(state_copy)

# get value of misplaced tile heuristic for a current state and goal state.
# misplaced tile distance is the count of all the tiles that are not in correct position
# includes blank for calculation
def misplaced_tile_heuristic(current_state, goal_state):
    misplaced_count = 0

    for i in range(len(goal_state)):
        if current_state[i] != goal_state[i]:
            misplaced_count += 1
    
    return misplaced_count

# generate random states for evaulating and testing
def generate_random_states(state_len):
    arr = [i for i in range(state_len)]
    np.random.shuffle(arr)
    return arr 

# get row and column positions for a given element in a given state
# used for manhattan distance
def get_row_col_position(state, element):

    # get total number of elements per row/column
    sqrt = int(math.sqrt(len(state)))

    idx = state.index(element)

    c = idx % sqrt
    r = idx // sqrt

    return r,c

# get value for manhattan distance for a current state and goal state
# manhattan distance is the shortest distance a tile needs to be moved to get to correct position
# total distance is sum of all individual distances
# includes blank for calculation
def manhattan_distance_heuristic(goal_state, current_state):
    total_manhattan_distance = 0

    # get total number of elements per row/column
    sqrt = int(math.sqrt(len(goal_state)))

    for i in goal_state:
        goal_state_row, goal_state_colums = get_row_col_position(goal_state, i)
        random_state_row, random_state_colums = get_row_col_position(current_state, i)

        total_manhattan_distance += abs(goal_state_colums - random_state_colums) + abs(goal_state_row - random_state_row)
    
    return total_manhattan_distance


# print out landing page.
# get algo choice
# get input state
# validate input
# call the search function
def main_block(clear_previous = True):

    puzzle_state = 8
    # clear screen before landing page
    if clear_previous:
        os.system('cls')
    
    # main block
    # Get algo choice
    print (f'---- {puzzle_state} puzzle solver ----')
    print ('1. Uniform Cost Search')
    print ('2. A* with Misplaced Tile')
    print ('3. A* with Manahattan Distance')
    algo_choice = int(input('Enter choice: '))

    if algo_choice not in [1,2,3]:
        os.system('cls')
        print ('Please enter correct choice.\n')
        main_block(clear_previous = False)
        return
        
    # get input puzzle state
    # TODO : Add code to choose between default puzzle and input state
    # TODO : Add code to let user input goal state as well
    print ('\nEnter the numbers in puzzle as space seperated list.')
    print ('Represent blank with 0')
    print ('For Example: 1 2 3 4 0 5 6 7 8\n')
    problem_input = input('Numbers: ')
    problem_state = problem_input.split(' ')

    # convert string to integers
    problem_state = [int(i) for i in problem_state]

    if not validate_numbers_input(problem_state, puzzle_state):
        os.system('cls')
        print ('Pleae enter valid input state.\n')
        main_block(clear_previous = False)
        return
    
    os.system('cls')
    print ('Initial State\n')
    print_state (problem_state, puzzle_state)

    if algo_choice == 1:
        print ('\nSolving for Uniform cost\n')
    elif algo_choice == 1:
        print ('\nSolving for A* with Misplaced Tile\n')
    elif algo_choice == 1:
        print ('\nSolving for A* with Manahattan Distance\n')

    return


main_block()