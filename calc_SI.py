# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:47:32 2023

@authors: Rohini, Lillian
"""

import numpy as np
import math
import os

def normalize_inputs(inputs, input_ranges):
    """
    This function normalizes the inputs given a numpy array of input ranges.
    :param inputs: the inputs to normalize
    :param input_ranges: the input ranges to normalize the inputs to

    :return: the normalized inputs
    """
    norm_inputs = np.zeros(np.shape(inputs))

    for i in range(np.shape(input_ranges)[1]):
        norm_inputs[:, i] = (inputs[:, i] - input_ranges[0,i]) / (input_ranges[1,i] - input_ranges[0,i])
    return norm_inputs

def reorganize_vars(M, N, K, policy):
    """ 
    This function reorganizes the policy variables into the centers, radii, and weights for each RBF.
    :param M: the number of input variables
    :param N: the number of RBF functions
    :param K: the number of outputs
    :param policy: the policy variables

    :return: the arrays of centers, radii, and weights for each RBF, in that order
    """
    C = np.zeros([M+2,N])
    B = np.zeros([M+2,N])
    W = np.zeros([K,N])

    # the policy is a combination of the centers, radii, and weights 
    # for each input for each RBF

    for n in range(N):  # for each RBF function
        for m in range(M):  # for each input 
            C[m,n] = policy[(2*M+K)*n + 2*m]  # get each center for each state variable (aka input)

            if policy[(2*M+K)*n + 2*m + 1] < 10**-6:   
                # get each radius for each state variable (aka input)
                B[m,n] = 10**-6   # if radius is too small, set to 10^-6  
            else:
                B[m,n] = policy[(2*M+K)*n + 2*m + 1]
        
        C[M,n] = 0.0
        C[M+1,n] = 0.0
        B[M,n] = 1.0
        B[M+1,n] = 1.0

        for k in range(K):
            W[k,n] = policy[(2*M+K)*n + 2*M + k]   # get the weights for each output (aka input)
    
    totals = np.sum(W,1)

    # normalize weights to sum to 1 across each RBF 
    # each row (for each output) should sum to 1
    for k in range(K):
        if totals[k] > 10**-6:
            W[k,:] = W[k,:]/totals[k]
    
    # return the centers, radii, and weights for each input for each RBF 
    return C, B, W

def calc_daily_data(soln, years, days_of_year, day, IO_ranges, phi1, phi2, M, K):
    """
    This function calculates the daily data for a given day of the year.
    :param soln: the solution number
    :param years: the number of years to simulate across
    :param day_of_year: the day of the year
    :param day: the day number
    :param IO_ranges: the input/output ranges of the decision variables and the release policies 
    :param phi1: the phase shift for the sine function
    :param phi2: the phase shift for the cosine function
    :param M: the number of input variables
    :param N: the number of RBF functions
    :param K: the number of outputs

    :return: the daily data for the given day

    It takes in the solution number, the number of years, the day of the year,
    the day number, the input/output ranges, the phase shifts, the number of
    inputs, the number of outputs, and the number of policy inputs.
    It outputs the daily data for the given day.
    """

    daily_data = np.loadtxt('HydroInfo_100_thinned_proc' + str(soln) + '_day' 
                            + str(days_of_year[day]) + '.txt')

    # convert each of the day per year to radians
    # no phase shift needed since cosine is a shifted sine
    sin_array = np.ones([years, 1]) * np.sin(2*math.pi * days_of_year[day]/365.0 - phi1)
    cos_array = np.ones([years, 1]) * np.cos(2*math.pi * days_of_year[day]/365.0 - phi2)

    # make the full policy input vector x_t
    # concatenate sin()
    daily_data = np.concatenate((np.concatenate((daily_data[:, 0:M], sin_array), 1), 
                                 daily_data[:, M:M+K]), 1) 
    #print("First concatenation: ", daily_data)

    daily_data = np.concatenate((np.concatenate((daily_data[:, 0:(M+1)], cos_array), 1), 
                                daily_data[:, (M+1):(M+1+K)]), 1)
    #print("Second concatenation: ", daily_data)

    daily_data = normalize_inputs(daily_data[:, 0:(M+2+K)], IO_ranges)
    #print("Normalized inputs: ", daily_data)
    return daily_data

def calc_deriv(C, B, W, M, N, input_num, input_vals, output_num):
    """
    This function calculates the derivative of the release relative to the input and RBF.
    :param C: the centers of the RBFs
    :param B: the radii of the RBFs
    :param W: the weights of the RBFs   
    :param M: the number of input variables
    :param N: the number of RBF functions
    :param input_num: the input index number
    :param input_vals: the input values
    :param output_num: the output index number

    :return: the derivative of the release relative to the input and RBF
    """

    deriv = 0
    for n in range(N):
        inner_sum = 0  ## initialize the inner sum and keep changing its number until all C, B are accounted for
        for m in range(M):
            inner_sum = inner_sum - (input_vals[m] - C[m, n])**2 / B[m, n]**2
        
        # Eqn 12 in Quinn et al 2019
        deriv = deriv - 2 * W[output_num, n] * ((input_vals[input_num] - C[input_num, n])/
                              (B[input_num, n])**2) * np.exp(inner_sum)
    return deriv

def calc_analytical_SI(M, N, K, policy_file, soln_num, num_years, inputNames, outputNames):
    """
    This function calculates the analytical sensitivity indices for a given solution.
    :param M: the number of input variables
    :param N: the number of RBF functions
    :param K: the number of outputs
    :param policy_file: the file containing the policy
    :param soln_num: the solution number
    :param num_years: the number of years to simulate across
    :param days: the number of days in a year
    :param inputNames: the names of the input variables to the model that might be driving variability
    :param outputNames: names of the model output where variability is being measured

    :return: None
    """

    # Get the Pareto-optimal policy (last two entries are the performance objectives)
    policy_dir = os.getcwd()
    print(policy_dir + '/' + policy_file)
    policy_vars = np.loadtxt(policy_dir + '/' + policy_file)  

    #print(len(policy_vars))
    soln = soln_num  ## take the first solution
    days = 365 ## number of days in a year
    years = num_years  ## number of years in the simulation 
    
    #input_names = inputNames  ## names of the input to the model that might be driving variability
    #output_names = outputNames  ## names of the model output where variability is being measured

    # Each row: input and output lower/upper bounds 
    # First 4 elements are inputs including bounds on sin() and cos()
    # Next 4 are outputs (releases)
    IO_ranges = np.array([[2223600000, 3215000000, 402300000, 402300000, 0, -1, -1, \
                           0, 0, 0, 0], \
                            [12457000000, 10890000000, 2481000000, 3643000000, 20, 1, 1, \
                             40002, 35784, 13551, 3650]])
    header = ''

    # Names of first order indices
    for input in inputNames:
        header = header + ',' + input + '_1'  ## eg: lake_level_1, p_fcast_1
    
    # Names of second order indices
    for i in range(len(inputNames) - 1):
        for j in range(i + 1, len(inputNames)):
            header = header + ',' + inputNames[i] + '+' + inputNames[j]  ## eg: lake_level+p_fcast
        
    header = header[1:]  ## remove beginning comma
    days_of_year = np.array(np.arange(1, 366, 1))  ## days of the year

    # load decision variables of this solution
    # Decision variables: C, B, and W + 1 constant alpha (Giuliani et al 2019)
    num_dvs = N*(K + 2*((M+2) - 2))  # should be 168, drawn from p.5698 in Quinn et al 2019)
    policy = policy_vars[soln_num-1, 0:num_dvs]  ## take the decision variables of the policy
    phi1 = [soln_num-1, num_dvs]
    phi2 = [soln_num-1, num_dvs+1]
    C, B, W = reorganize_vars(M, N, K, policy)  ## what is this function doing?
    
    ## change directory to the daily folder
    change_dir = 'daily/Soln_' + str(soln) + '/'
    os.chdir(change_dir)  

    # find covariances at each day
    cov = np.zeros([days, M, M])  # initialize covariance matrix between each reservoir for each day
    all_data = np.zeros([years, days, M+2+K]) # all data at each day across all years (include both input and output, phi1 and phi2)

    for day in range(days):
        daily_data = calc_daily_data(soln, years, days_of_year, day, IO_ranges, phi1, phi2, M, K)
        all_data[:, day, :] = daily_data   # calculate the daily data for I/O of one day across all years
        cov[day, :, :] = np.cov(np.transpose(daily_data[:, 0:M]))  # store the transpose as the covariance matrix

    # find the SI at each time step 
    # TO DO: section can be vectorized?
    for output in range(K):   # at each reservoir
        # all_SI and all_D have size (years * days, M + (M*(M-1))/2) = (4745, 10)
        all_SI = np.zeros([days * years, int(M + (M*(M-1))/2)])
        #all_var = np.zeros([days * years, int((M-2) + (M-2)*(M-3)/2)])

        for year in range(years):  # for each year
            for day in range(days):  # for each day in year
                input_vals = all_data[year, day, 0:M+2]  # get the daily precip, lake level forecasts
                
                for col in range(M):  # get the first order indices
                    # calculate the derivative of release relative to input and RBF
                    deriv = calc_deriv(C, B, W, M, N, col, input_vals, output) 
                    
                    # Eqn 10 in Quinn et al 2019
                    # Use the deriv to calculate the first order SI and store the deriv
                    all_SI[year*days + day, col] = deriv**2 * cov[day, col, col]
                    #all_var[year*days + day, col] = cov[day, col, col]
                
                count = 0
                for col1 in range(M-1):
                    for col2 in range(col1+1, M):
                        # Use the deriv to calculate the second order SI
                        D1 = calc_deriv(C, B, W, M, N, col1, input_vals, output)
                        D2 = calc_deriv(C, B, W, M, N, col2, input_vals, output)

                        # Eqn 11 in Quinn et al 2019
                        all_SI[year*days + day, M+count] = 2*D1*D2*cov[day, col1, col2]
                        #all_var[year*days + day, M+count-2] = cov[day, col1, col2]
                        count = count + 1
        
        filename_dir = "/release_decisions/"

        #main_folder = os.path.dirname(os.path.dirname(change_dir))
        #print('main folder: ', main_folder)
        #change_dir = main_folder + '/sensitivities/'
        #os.chdir(change_dir)
        
        #if os.path.exists("Soln_" + str(soln_num) + "/") == False:
            #os.makedirs("Soln_" + str(soln_num) + "/")
        
        if os.path.exists(filename_dir) == False:
            os.makedirs("/release_decisions/")
            os.makedirs("/Figures/")
        
        filename_soln = filename_dir + outputNames[output]  + ".txt"
        #filename_deriv = "Solution_precip_var_" + str(output) + ".txt"
        
        np.savetxt(filename_soln, all_SI, comments='', header=header, delimiter=",")
        #np.savetxt(filename_deriv, all_var, comments='', header=header, delimiter=",")
        print(f'Saved sensitivity indices for {outputNames[output]}')
    
    return None