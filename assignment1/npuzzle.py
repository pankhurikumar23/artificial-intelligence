"""
Artificial Intelligence - Programming Homework 1

Implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size.

@author: PANKHURI KUMAR (PK2569)
"""

import time

def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped.
    """
    # print(state)
    value1 = state[i1][j1]
    value2 = state[i2][j2]

    new_state = []
    for row in range(len(state)):
        new_row = []
        for column in range(len(state[row])):
            if row == i1 and column == j1:
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else:
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)


def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions.
    The result is a list containing (Action, state) tuples.
    For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))),
                 ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))]
    """

    child_states = []
    m = len(state)
    n = len(state[0])
    row = 0
    col = 0
    # print(m, n)

    # print("In G_S:")
    # print(state_to_string(state))

    #find position of blank (0) on board
    flag = 0
    for r in range(m):
        for c in range(n):
            if state[r][c] == 0:
                flag = 1
                row = r
                col = c
                break
        if flag == 1:
            break

    #shift tiles based on position of blank
    #checking for boundary case
    new_tuple = []
    if (col+1 < n):
        new_tuple.append("Left")
        new_tuple.append(swap_cells(state, row, col, row, col+1))
        child_states.append(tuple(new_tuple))
        new_tuple = []
    if (col-1 > -1):
        new_tuple.append("Right")
        new_tuple.append(swap_cells(state, row, col, row, col-1))
        child_states.append(tuple(new_tuple))
        new_tuple = []
    if (row+1 < m):
        new_tuple.append("Up")
        new_tuple.append(swap_cells(state, row, col, row+1, col))
        child_states.append(tuple(new_tuple))
        new_tuple = []
    if (row-1 > -1):
        new_tuple.append("Down")
        new_tuple.append(swap_cells(state, row, col, row-1, col))
        child_states.append(tuple(new_tuple))

    return child_states


def create_goal_state(n):
    """
	Returns the goal state for a n-puzzle.
	"""
    goalState = []
    val = 0

    for r in range(n):
        new_row = []
        for c in range(n):
            new_row.append(val)
            val += 1
        goalState.append(tuple(new_row))

    return tuple(goalState)

def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise.
    """
    n = len(state)
    goalState = create_goal_state(n)

    # print(goalState)
    return (state == goalState)

def recover_solution(goalState, initialState, parents, actions):
    """
    Recovers the sequence of actions taken from initial to goal state.
    """
    sol = []
    state = goalState
    # print(goalState)
    while state != initialState:
        # print(parents[state])
        sol.append(actions[state])
        state = parents[state]

    sol.reverse()
    # print(sol)
    return sol


def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    parents = {}
    actions = {}
    solution = []

    states_expanded = 0
    max_frontier = 0

    frontier = [state]
    explored = set()
    seen = set()
    seen.add(state)

    #BFS algorithm
    while len(frontier) != 0:
        node = frontier.pop(0)
        explored.add(node)
        states_expanded += 1

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

        if goal_test(node):
            # print("Success. Need to call function here.")
            solution = recover_solution(node, state, parents, actions)
            # print(states_expanded)
            break

        successors = get_successors(node)
        for successor in successors:
        	#separating action (left, right,...) and successor state
            nextState = successor[1]
            action = successor[0]
            if nextState not in explored and nextState not in seen:
                # print(state_to_string(s[1]) + '\n')
                parents[nextState] = node
                actions[nextState] = action
                frontier.append(nextState)
                seen.add(nextState)

    #  return solution, states_expanded, max_frontier
    return solution, states_expanded, max_frontier


def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """

    parents = {}
    actions = {}
    solution = []

    states_expanded = 0
    max_frontier = 0

    frontier = [state]
    explored = set()
    seen = set()
    seen.add(state)

    #DFS algorithm
    while len(frontier) != 0:
        # print(parents)
        node = frontier.pop()
        explored.add(node)
        states_expanded += 1

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

        if goal_test(node):
            # print("Success. Need to call function here.")
            solution = recover_solution(node, state, parents, actions)
            # print(solution)
            break

        successors = get_successors(node)
        for successor in successors:
            nextState = successor[1]
            action = successor[0]
            if nextState not in explored and nextState not in seen:
                # print(state_to_string(s[1]) + '\n')
                parents[nextState] = node
                actions[nextState] = action
                frontier.append(nextState)
                seen.add(nextState)


    #  return solution, states_expanded, max_frontier
    return solution, states_expanded, max_frontier


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """
    misplaced = 0
    n = len(state)
    goalState = create_goal_state(n)
    subtractedTuple = []
    #to account for 0
    if state[0][0] != 0:
        misplaced = -1

    # How to subtract tuples taken from User Jared: https://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
    from operator import sub
    for row in range(n):
        subtractedRow = tuple(map(sub, state[row], goalState[row]))
        subtractedTuple.append(subtractedRow)
    # print(subtractedTuple)

    for row in range(n):
        for col in range(n):
            if subtractedTuple[row][col] != 0:
                misplaced += 1

    return misplaced


def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the manhattan distance between the current
    position and the goal position. THen sum all distances.
    """
    n = len(state)
    manhattan = 0

    for row in range(n):
        for col in range(n):
            value = state[row][col]
            if value != 0:
                # i & j denote the goal position of the value
                i = int(value/n)
                j = value%n
                # distance is equivalent to difference of indexes
                manhattan += (abs(row - i) + abs(col - j))

    # print(manhattan)
    return manhattan


def best_first(state, heuristic = misplaced_heuristic):
    """
    Breadth first search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    from heapq import heappush
    from heapq import heappop

    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0
    solution = []

    states_expanded = 0
    max_frontier = 0

    frontier = [(costs[state], state)]
    explored = set()

    #GBFS algorithm
    while len(frontier) != 0:
        #node is (cost, state) format
        node = heappop(frontier)
        poppedState = node[1]
        explored.add(poppedState)
        states_expanded += 1

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

        if goal_test(poppedState):
            solution = recover_solution(poppedState, state, parents, actions)
            break

        successors = get_successors(poppedState)
        # data = []
        #successor in (action, state) format
        for successor in successors:
            nextState = successor[1]
            action = successor[0]
            cost = heuristic(nextState)
            if nextState not in explored and (cost, nextState) not in frontier:
                parents[nextState] = poppedState
                actions[nextState] = action
                costs[nextState] = cost
                heappush(frontier, (cost, nextState))

        # for item in data:
        #     heappush(frontier, item)


    #  return solution, states_expanded, max_frontier
    return solution, states_expanded, max_frontier

def astar(state, heuristic = misplaced_heuristic):
    """
    A-star search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    from heapq import heappush
    from heapq import heappop

    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0
    solution = []

    states_expanded = 0
    max_frontier = 0

    frontier = [(costs[state], state)]
    explored = set()

    #A* algorithm
    while len(frontier) != 0:
        node = heappop(frontier)
        poppedState = node[1]
        explored.add(poppedState)
        states_expanded += 1

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

        if goal_test(poppedState):
            solution = recover_solution(poppedState, state, parents, actions)
            break

        successors = get_successors(poppedState)
        # successor in (action, state) format
        for successor in successors:
            nextState = successor[1]
            action = successor[0]
            #value for h(n) in f(n) = g(n)+h(n)
            h_value = heuristic(nextState)
            #costs stores value of only g(n)
            cost = costs[poppedState] + 1
            if nextState not in costs:
                parents[nextState] = poppedState
                actions[nextState] = action
                costs[nextState] = cost
                heappush(frontier, (cost+h_value, nextState))
            elif cost < costs[nextState]:
                f_value = costs[nextState] + h_value

                # How to delete particular node from heap taken from Duncan:
                # https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
                index = frontier.index((f_value, nextState))
                frontier[index] = frontier[-1]
                x = frontier.pop()

                costs[nextState] = cost
                parents[nextState] = poppedState
                actions[nextState] = action
                heappush(frontier, (cost+h_value, nextState))

    # print(frontier)
    return solution, states_expanded, max_frontier


def print_result(solution, states_expanded, max_frontier):
    """
    Helper function to format test output.
    """
    if solution is None:
        print("No solution found.")
    else:
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max frontier size: {}.".format(max_frontier))



if __name__ == "__main__":

    #Easy test case
    test_state = ((1, 4, 2),
                  (0, 5, 8),
                  (3, 6, 7))

    #More difficult test case
    # test_state = ((7, 2, 4),
    #              (5, 0, 6),
    #              (8, 3, 1))

    print(state_to_string(test_state))
    print()

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_frontier = bfs(test_state) #
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====DFS====")
    start = time.time()
    solution, states_expanded, max_frontier = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====A* (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====A* (Total Manhattan Distance Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

