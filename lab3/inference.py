import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 

    for i in range(num_time_steps):
        # TODO: Compute the forward messages
        forward_messages[i] = rover.Distribution()
        # Intialization
        if i == 0:
            for state in prior_distribution:
                result = prior_distribution[state] * observation_model(state)[observations[0]]
                if result != 0:
                    forward_messages[i][state] = result
        # Recursive
        else:
            for state in all_possible_hidden_states:
                result = 1
                if observations[i] != None:
                    result = observation_model(state)[observations[i]]
                subresult = 0
                for k in forward_messages[i-1]:
                    product = transition_model( k)[state] * forward_messages[i-1][k]
                    subresult += product
                result *= subresult
                if result != 0:
                    forward_messages[i][state] = result
        forward_messages[i].renormalize() 

        # TODO: Compute the backward messages
        n = num_time_steps - i - 1
        backward_messages[n] = rover.Distribution()
        # Intialization
        if i == 0:
            for state in all_possible_hidden_states:                
                backward_messages[n][state] = 1
        # Recursive
        else:
            for state in all_possible_hidden_states:
                result = 0
                subresult = 1
                for k in backward_messages[n+1]:
                    subresult = 1
                    if observations[n+1] != None:
                        subresult = observation_model(k)[observations[n+1]]
                    subresult*= transition_model(state)[k] * backward_messages[n+1][k]
                    result += subresult
                if result != 0:
                    backward_messages[n][state] = result
        backward_messages[n].renormalize() 

    # TODO: Compute the marginals 
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for state in all_possible_hidden_states:
            result = forward_messages[i][state] * backward_messages[i][state]
            if result != 0:
                marginals[i][state] = result
        marginals[i].renormalize()
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    w = [None] * num_time_steps
    backtrack = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    for i in range(num_time_steps):
        w[i] = rover.Distribution()
        # Initialization
        if i == 0:
            for state in prior_distribution:
                result = prior_distribution[state]
                result *= observation_model(state)[observations[0]]
                if result != 0:
                    w[i][state] = np.log(result)
        # Recursive
        else:
            backtrack[i] = {}
            for state in all_possible_hidden_states:
                p1 = 1
                if observations[i] != None:
                    p1 = observation_model(state)[observations[i]]
                if p1 == 0:
                    continue
                p2 = - np.inf
                for k in w[i - 1]:
                    if transition_model(k)[state] == 0:
                        continue
                    temp = np.log(transition_model(k)[state]) + w[i - 1][k]
                    if temp > p2:
                        p2 = temp
                        backtrack[i][state] = k
                result = np.log(p1) + p2
                if p2 != -np.inf and result != 0:
                    w[i][state] = result
    
    # Backtracking
    n = num_time_steps - 1
    estimated_hidden_states[n] = max(w[n] , key = w[n].get )
    for i in range(1, num_time_steps):
        n = num_time_steps - 1 -i
        k = estimated_hidden_states[n + 1 ]
        estimated_hidden_states[n] = backtrack[n + 1][k]
    
    return estimated_hidden_states



if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n') 

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
    
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps

    v_count = 0
    for i in range(len(estimated_states)):
        if estimated_states[i] == hidden_states[i]:
            v_count += 1
    error_v = 1 - v_count/100

    fb_count = 0

    for i in range(len(marginals)):
        state = max(marginals[i] , key = marginals[i].get)
        if state == hidden_states[i]:
            fb_count += 1
    error_fb = 1 - fb_count /100

    print("Vertibi Error:" , error_v)
    print("forward_backward Error:" , error_fb)

    count = 0
    for i in range(len(marginals)):
        state = max(marginals[i] , key = marginals[i].get)
        # print(count , ") ",state)
        count += 1



    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
