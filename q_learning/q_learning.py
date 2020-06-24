"""
The original code is from https://github.com/dennybritz/reinforcement-learning/tree/master/TD
"""

import sys
import numpy as np
import itertools
import pickle
from collections import defaultdict
from game import Game

# In our case, we have 3 action (stay, go-left, go-right)
def get_action_num():
    return 3


## this function return policy function to choose the action based on Q value.
def make_policy(Q, epsilon, nA):
    """
    This is the epsilon-greedy policy, which select random actions for some chance (epsilon).
    (Check dennybritz's repository for detail)

    You may change the policy function for the given task.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

	
## this function return state from given game information.
def get_state(counter, score, game_info):
    basket_location, item_location = game_info
    """
    FILL HERE!
    you can
    (and must) change the return value.
    """
    itemnum = len(item_location) #number of item
    check=False #bool variable for item check
    check2=False #bool varialbe for item check
    movement=0 #result of function
    time=0 #find index of time item
    item=0 #find index of item
    for j in range (0, itemnum): #find time item function
        name=item_location[j][0]
        if(name==2):
            time=j
            check2=True
            break
    if(check2==True):#There is time item
        check2=True
    else:#There is not time item
        for i in range (0, itemnum):
            x=round(item_location[i][1]*7/11)
            y=10-item_location[i][2]
            if(y-abs(basket_location-x) > 0):
                item=i
                check=True #Find can go
                break
    if(check==False and check2 == False): #Can't find can go
        if(basket_location>4):
            movement=1
        if(basket_location<4):
            movement=2
        if(basket_location==4):
            movement=0
    if(check==False and check2==True): #go to time item
        if(basket_location>round(item_location[time][1]*7/11)):
            movement=1
        if(basket_location<round(item_location[time][1]*7/11)):
            movement=2
        if(basket_location==round(item_location[time][1]*7/11)):
            movement=0
    if(check==True and check2==False): #go to score item
        if(basket_location>round(item_location[item][1]*7/11)):
           movement=1
        if(basket_location<round(item_location[item][1]*7/11)):
           movement=2
        if(basket_location==round(item_location[item][1]*7/11)):
           movement=0
           
    return movement
           
## this function return reward from given previous and current score and counter.
def get_reward(prev_score, current_score, prev_counter, current_counter):
    """
    FILL HERE!
    you can (and must) change the return value.
    """
    re = (current_score - prev_score) + (current_counter - prev_counter) #reward is same as original code in general
    if(current_score == 0 or prev_score == 0): #when game finished, reward will be -500
        re = -500
    elif(current_counter < 200 or prev_counter < 200):
        re=(current_score - prev_score) + (current_counter - prev_counter)*10
    return re

def save_q(Q, num_episode, params, filename="model_q.pkl"):
    data = {"num_episode": num_episode, "params": params, "q_table": dict(Q)}
    with open(filename, "wb") as w:
        w.write(pickle.dumps(data))

        
def load_q(filename="model_q.pkl"):
    with open(filename, "rb") as f:
        data = pickle.loads(f.read())
        return defaultdict(lambda: np.zeros(3), data["q_table"]), data["num_episode"], data["params"]


def q_learning(game, num_episodes, params):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    You can edit those parameters, please speficy your changes in the report.
    
    Args:
        game: Coin drop game environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
    """
    
    epsilon, alpha, discount_factor = params
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(get_action_num()))  
    
    # The policy we're following
    policy = make_policy(Q, epsilon, get_action_num())
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        _, counter, score, game_info = game.reset()
        state = get_state(counter, score, game_info)
        action = 0
        # One step in the environment
        for t in itertools.count():
            # Take a step
            action_probs = policy(get_state(counter, score, game_info))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            done, next_counter, next_score, game_info = game.step(action)
            
            next_state = get_state(counter, score, game_info)
            reward = get_reward(score, next_score, counter, next_counter)
            counter = next_counter
            score = next_score
            
            """
            this code performs TD Update. (Update Q value)
            You may change this part for the given task.
            """
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state 
        
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("Episode {}/{} (Score: {})\n".format(i_episode + 1, num_episodes, score), end="")
            sys.stdout.flush()

    return Q

def train(num_episodes, params):
    g = Game(False)
    Q = q_learning(g, num_episodes, params)
    return Q


## This function will be called in the game.py
def get_action(Q, counter, score, game_info, params):
    epsilon = params[0]
    policy = make_policy(Q, epsilon, 3)
    action_probs = policy(get_state(counter, score, game_info))
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episode", help="# of the episode (size of training data)",
                    type=int, required=True)
    parser.add_argument("-e", "--epsilon", help="the probability of random movement, 0~1",
                    type=float, default=0.1)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of training",
                    type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.num_episode is None:
        parser.print_help()
        exit(1)

    # you can pass your parameter as list or dictionary.
    # fix corresponding parts if you want to change the parameters
    
    num_episodes = args.num_episode
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    
    Q = train(num_episodes, [epsilon, learning_rate, 0.5])
    save_q(Q, num_episodes, [epsilon, learning_rate, 0.5])
    
    #Q, n, params = load_q()
    
if __name__ == "__main__":
    main()
