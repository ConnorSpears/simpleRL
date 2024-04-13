
import numpy as np
import pandas as pd
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os

np.random.seed(2)  # reproducible

N_STATES = 10  # the size of the 2 dimensional world
ACTIONS = ['left', 'right', 'up', 'down']     # available actions
EPSILON = 0.4   # greedy police
EPSILON_DECAY = 0.014  # Adjust this rate to control the decay speed
MAX_EPSILON = 0.999 #Maximum value of epsilon (min exploration rate is 1%)
ALPHA = 0.06     # learning rate
GAMMA = 0.8   # discount factor
MAX_EPISODES = 1000   # maximum episodes
LAMBDA = 0.95  # Eligibility trace decay factor
FRESH_TIME = 0.3   # fresh time for one move



def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states*n_states, len(actions))),     # q_table initial values
        columns=actions,    
    )

    return table

def build_eligibility_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states*n_states, len(actions))),  # Eligibility trace table
        columns=actions,
    )
    return table

#Maze generator that creates a random maze with a fairly direct path from start to goal

# def generate_maze(n_states):
#     #Generate an empty maze
#     maze = []

#     #Ensure there's a path from start to goal
#     path = set()
#     x, y = 0, 0
#     while x < n_states - 1 or y < n_states - 1:
#         if np.random.rand() > 0.5 and x < n_states - 1:
#             x += 1
#         elif y < n_states - 1:
#             y += 1
#         path.add((x, y))

#     #Add obstacles randomly
      #Need to tune this number depending on n_states !!!
#     num_obstacles = n_states * 10

#     while len(maze) < num_obstacles:
#         obstacle = (np.random.randint(0, n_states), np.random.randint(0, n_states))
#         if obstacle not in path and obstacle != (0, 0) and obstacle not in maze:
#             maze.append(obstacle)

#     return maze




#Maze generator that uses a dfs to create a long, winding path from start to goal

def generate_maze(n_states):
    # Initialize the grid, marking all cells as obstacles initially
    maze = -np.ones((n_states, n_states), dtype=int)
    visited = np.zeros((n_states, n_states), dtype=bool)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Possible movements: down, right, up, left

    obstacles = []

    def make_path(x, y):
        if x == n_states - 1 and y == n_states - 1:  # If we've reached the bottom right corner
            visited[x, y] = True
            maze[x, y] = 0  # Mark as part of the path
            return True

        if not (0 <= x < n_states) or not (0 <= y < n_states) or visited[x, y]:
            return False  # Out of bounds or already visited

        visited[x, y] = True  # Mark as visited
        maze[x, y] = 0  # Mark as part of the path temporarily

        np.random.shuffle(directions)  # Shuffle directions to randomize the path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if make_path(nx, ny):
                return True  # Path successful

        maze[x, y] = -1  # Backtrack: mark as obstacle if no path forward
        return False

    make_path(0, 0)  # Start pathing from the top-left corner

    #add some random path cells
    for i in range(n_states):
        for j in range(n_states):
                if np.random.rand() < 0.2:
                    maze[i, j] = 0

                if maze[i,j]==-1:
                    obstacles.append((i,j))
                    

    # Ensure start and goal are clear
    maze[0, 0] = 0  # Start
    maze[n_states - 1, n_states - 1] = 0  # Goal

    # print(maze)
    # time.sleep(5)
    # print(obstacles)
    # time.sleep(5)

    return obstacles






def choose_action(X,Y, q_table):

    _S = X*N_STATES+Y

    state_actions = q_table.iloc[_S, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
        is_exploratory = True
    else:   # act greedy
        action_name = state_actions.idxmax()    
        is_exploratory = False
    return action_name, is_exploratory


def get_env_feedback(X,Y, A,maze):
    # This is how agent will interact with the environment
    X_ = X
    Y_ = Y

    if A == 'right':    
        if (X == N_STATES - 1):
          X_ = X
          R = 0

        elif (X == N_STATES - 2) and (Y == N_STATES-1):
          X_ = 'terminal'
          R = 1
        else:
            X_ = X + 1
            R = 0

    if A == 'left':
        R = 0
        if X == 0:
            X_ = X  
        else:
            X_ = X-1

    if A == 'up':    
        if Y == 0:   
            Y_ = Y
            R = 0
        else:
            Y_ = Y - 1
            R = 0

    if A == 'down':
        if Y == N_STATES - 1:
          Y_ = Y
          R = 0
        elif (Y == N_STATES - 2) and (X == N_STATES - 1):
          X_ = 'terminal'
          R=1

        else:
            Y_ = Y + 1
            R=0

    #check if you land on an obstacle

    if (Y_, X_) in maze:
      R = -2

    return X_,Y_, R

def update_env(X, Y, episode, step_counter,maze):

    clear_output()
    os.system('clear')

    if X == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)

    else:
        grid_representation = ""  # Initialize an empty string to accumulate grid representation
        for i in range(N_STATES):
            env_list = ['-'] * (N_STATES)  # Adjusted for N_STATES length to include 'T'
            if i == N_STATES - 1:  # Place 'T' at the rightmost location of the last line
                env_list[-1] = 'T'

            #place the obstacles

            for k in range(N_STATES):
              if (i,k) in maze:
                env_list[k] = 'X'


            if i == Y:  # Agent is represented by 'o'
                env_list[X] = 'o'

            interaction = ''.join(env_list) + "\n"  # Add newline to separate rows
            grid_representation += interaction  # Accumulate the representation

        print(grid_representation, end='')  # Print the accumulated grid representation
        time.sleep(FRESH_TIME)


def rl():
    global EPSILON, EPSILON_DECAY
    steps_over_time = []
    q_table = build_q_table(N_STATES, ACTIONS)
    e_table = build_eligibility_table(N_STATES, ACTIONS)
    maze = generate_maze(N_STATES)

    for episode in range(MAX_EPISODES):
        step_counter = 0
        X, Y = 0, 0
        is_terminated = False

        if episode%10 == 0:
          #Decrease exploitation rate
          EPSILON = min(MAX_EPSILON, EPSILON + (1-EPSILON) * EPSILON_DECAY)
          #Increase rate of decay
          EPSILON_DECAY = EPSILON_DECAY*1.01

        if episode%100 == 0:
          clear_output()
          os.system('clear')
          print(episode)

        if episode>=998:
          update_env(X,Y, episode, step_counter,maze)

        while not is_terminated:
            S = X * N_STATES + Y
            A, is_exploratory = choose_action(X, Y, q_table)
            X_, Y_, R = get_env_feedback(X, Y, A, maze)

            if X_ != 'terminal':
                S_ = X_ * N_STATES + Y_
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_predict = q_table.loc[S, A]
            error = q_target - q_predict

            # Eligibility trace update for the action taken
            e_table.loc[S, A] += 1

            # Q-table update
            q_table += ALPHA * error * e_table

            # print("State: ", X, Y)
            #print("Action taken: ", A)
            # print("Reward: ", R)
            # print("e_table: ")
            # print(e_table)
            # print("q_table: ")
            # print(q_table)


            # Manage eligibility traces
            if is_exploratory:
                e_table = build_eligibility_table(N_STATES, ACTIONS)  # Reset traces after exploratory move
            else:
                e_table *= GAMMA * LAMBDA  # Decay eligibility traces for all state-action pairs

            X, Y = X_, Y_
            step_counter += 1

            if episode>=998:
                update_env(X,Y, episode, step_counter,maze)

        steps_over_time.append(step_counter)

    return q_table, steps_over_time




if __name__ == "__main__":
    q_table, steps_over_time = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    plt.plot(steps_over_time)
    plt.xlabel("episode")
    plt.ylabel("steps")
    plt.show()

