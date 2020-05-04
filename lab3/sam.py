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
        forward_messages[i] = rover.Distribution({})  
        if i == 0:
            for zn in prior_distribution:
                z0 = observations[i]
                forward_messages[i][zn] = prior_distribution[zn] * observation_model(zn)[z0]
            forward_messages[i].renormalize()
        else:
            xn = observations[i]
            for zn in all_possible_hidden_states:
                if xn == None:
                    p_cond = 1
                else:
                    p_cond = observation_model(zn)[xn]
                if p_cond != 0:
                    prob = 0
                    for zn_prev in (forward_messages[i-1]):
                        prob += forward_messages[i-1][zn_prev] * transition_model(zn_prev)[zn]
                    if prob != 0:
                        forward_messages[i][zn] = p_cond * prob
            forward_messages[i].renormalize()
        #print(forward_messages[i], "for i = ",i)

        # TODO: Compute the backward messages
        end = num_time_steps - 1 - i
        backward_messages[end] = rover.Distribution({})
        if i == 0:
            for zn in all_possible_hidden_states:
                backward_messages[end][zn] = 1
        else:
            xn_aft = observations[end + 1]
            for zn in all_possible_hidden_states:
                prob = 0
                for zn_aft in backward_messages[end+1]:
                    if xn_aft == None:
                        p_cond = 1
                    else:
                        p_cond = observation_model(zn_aft)[xn_aft]
                    prob += backward_messages[end + 1][zn_aft] * transition_model(zn)[zn_aft] * p_cond
                if prob != 0:
                    backward_messages[end][zn] = prob
            backward_messages[end].renormalize()

    # TODO: Compute the marginals

    for i in range(num_time_steps):

        marginals[i] = rover.Distribution()
        for zn in all_possible_hidden_states:
            alpha = forward_messages[i][zn]
            beta = backward_messages[i][zn]
            if alpha * beta != 0:
                marginals[i][zn] = (alpha * beta)
        marginals[i].renormalize()
    print("Marginal for is",marginals[1], "for i = 1")
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
    num_time_steps = len(observations)
    forward_pass = [None] * num_time_steps
    w = [None] * num_time_steps
    max_w_zn = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    #Initialization
    w[0] = rover.Distribution()
    for zn in prior_distribution:
        prior_z0 = prior_distribution[zn]
        z0 = observations[0]
        pcond_X0_z0 = observation_model(zn)[z0]
        if prior_z0*pcond_X0_z0 != 0:
            w[0][zn] = np.log(prior_z0 * pcond_X0_z0)
    
    for i in range(1,num_time_steps):
        w[i] = rover.Distribution()
        xn = observations[i]
        max_w_zn[i] = dict()
        for zn in all_possible_hidden_states:
            # Part 2: check if observation is None
            if xn == None:
                # set conditional probability to be 1
                pcond_Xn_zn = 1
            else:
                pcond_Xn_zn = observation_model(zn)[xn]
            if pcond_Xn_zn != 0:
                # set maximum value to be -inf, and find the maximum of all values
                max_zn_prev = - np.inf
                # iterate through all the previous zn, find the max value, and save it
                for zn_prev in (w[i - 1]):
                    pcond_zn_zn_prev = transition_model(zn_prev)[zn]
                    # avoid log of 0 case
                    if pcond_zn_zn_prev != 0:
                        w_prev = w[i - 1][zn_prev]
                        new_zn_prev = np.log(pcond_zn_zn_prev) + w_prev
                        if new_zn_prev > max_zn_prev:
                            max_zn_prev = new_zn_prev
                            # store the maximum path to get to a certain zn from zn_prev
                            max_w_zn[i][zn] = zn_prev
                # set the zn to be ln (p( (xn,yn)| zn )) + max {ln(p(zn,zn_prev)) +  w(zn_prev)}
                if max_zn_prev != -np.inf:
                    w[i][zn] = np.log(pcond_Xn_zn) + max_zn_prev
    print(max_w_zn)
     # loop through all the time_steps to determine optimal path
    for i in range(num_time_steps):
        end = num_time_steps - 1 - i
        if i == 0:
            # find the most probable zn at the end
            max_zn = - np.inf
            zn_star = None
            for zn in w[end]:
                zn_temp = zn
                max_zn_temp = w[end][zn_temp]
                if max_zn_temp > max_zn:
                    max_zn = max_zn_temp
                    zn_star = zn_temp
            estimated_hidden_states[end] = zn_star
        else:
            # find the path the links the old zn_star to the current time
            zn_star_prev = estimated_hidden_states[end + 1]
            estimated_hidden_states[end] = max_w_zn[end + 1][zn_star_prev]
    
    
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
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


   
    timestep = num_time_steps - 1
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
    
    # print("Last 10 hidden states in the MAP estimate:")
    # for time_step in range(num_time_steps - 10, num_time_steps):
    #     print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
