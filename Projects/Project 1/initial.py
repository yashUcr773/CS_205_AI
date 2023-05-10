# for square root
import math

# clear output screen
import os 

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
def validate_numbers_input(problem_state, puzzle = 8):

    # generate valid inputs.
    # valid inputs range from 0 - n for n puzzle
    valid_inputs = [i for i in range(puzzle+1)]

    # put the problem state in a set to remove duplicates.
    problem_state_set = set(problem_state)

    # if the length of inputs is less than puzzle size,
    # input is invalid.
    if len(problem_state_set) != puzzle+1:
        return False

    for i in problem_state_set:
        if i not in valid_inputs:
            return False
    
    return True


# takes in list of numbers and print it in puzzle view
def print_state(problem_state, puzzle = 8):

    if validate_numbers_input(problem_state, puzzle):
        
        sqrt = int(math.sqrt(puzzle+1))

        if sqrt*sqrt != puzzle+1:
            raise Exception (f'Enter Valid Puzzle Format, {puzzle} is not valid')
        else:
            for i in range(len(problem_state)):
                print (problem_state[i], end = " ")
                if (i+1) % sqrt == 0:
                    print ()
    else:
        raise Exception ('Enter Valid Puzzle State')


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