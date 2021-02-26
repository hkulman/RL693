#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import gym
import tools
env = gym.make('Taxi-v2')
env.reset()
env.render()
if not hasattr(env, 'nb_states'):  env.nb_states = env.env.nS
if not hasattr(env, 'nb_actions'): env.nb_actions = env.env.nA
if not hasattr(env, 'model'):      env.model = env.env.P


# In[10]:


def policyIteration(env, gamma, theta):
    # 1. Initialization
    V = np.zeros(env.nb_states) # Initialize V to the number of states (500)
    pi = np.zeros(env.nb_states, dtype=int)  # Each V willl have a corresponding policy
    while True:
        # 2. Policy Evaluation
        while True:
            delta = 0 
            for s in range(env.nb_states): # iterate through states
                v = V[s] # keeping track of old V.
                V[s] = state_action_value(env, V=V, s=s, a=pi[s], gamma=gamma) 
                delta = max(delta, abs(v - V[s])) 
            if delta < theta: break 

        # 3. Policy Improvement
        policy_stable = True # Setting stability flag to true
        for s in range(env.nb_states): #Iterate over each state
            old_action = pi[s] # Save the old action.
            pi[s] = np.argmax([state_action_value(env, V=V, s=s, a=a, gamma=gamma)  # list comprehension
                            for a in range(env.nb_actions)]) # calculate maximum action for the current action given the state transition probabilities, rewards, and current estimate of the value function 
            if old_action != pi[s]: policy_stable = False # If the old action is different than the current action then the policy isn't stable
        if policy_stable: break # if the policy is stable then return the estimate of the value function and policy
    
    return V, pi


# In[11]:


def state_action_value(env, V, s, a, gamma):
    statevalue = 0  # state value for state s
    for p, s_, r, _ in env.model[s][a]: 
        statevalue += p * (r + gamma * V[s_]) # reward * discounted factor * value function of future states
        return statevalue


# In[12]:


V, pi = policyIteration(env, gamma=.9, theta=1e-8)


# In[14]:


OptimalValue = np.array(V)
OptimalValue


# In[16]:


a2w = {0:'S', 1:'N', 2:'E', 3:'W', 4:'PickUp', 5: 'DropOff'}
policy_arrows = np.array([a2w[x] for x in pi])
print(np.array(policy_arrows).reshape([-1, 5]))


# In[ ]:




