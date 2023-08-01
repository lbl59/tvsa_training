# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:45:03 2023

@author: Rohini and Lillian
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

sns.set_style("dark")

def averageSensitivity(input1, num_years):
    '''
    This function takes in a sensitivity array and averages the sensitivity across 1,000 years.
    :param input1: Sensitivity numpy array
    :param num_years: Number of years to average over

    :return: Averaged transposed sensitivity array
    '''
    input1_reshaped = np.reshape(input1, (num_years,365))
    avg_1 = np.mean(input1_reshaped, axis = 0)

    return np.transpose(avg_1)

def plot_tvsa_policy(num_years, soln_num, input_list, colors_list, output_name, title):
    '''
    This function plots the TVSA of a policy for a given solution number.
    :param num_years: Number of years
    :param soln_num: Solution number
    :param input_list: List of input variable names, including their interactions
    :param colors_list: List of colors to use for each input variable
    :param output_name: Name of the output (reservoir release) to plot
    :param title: Title of the plot

    :return: None
    '''
    colors = colors_list

    legend_labels = []

    for l in range(len(input_list)):
        input = plt.Rectangle((0,0), 1, 1, fc=colors[l], edgecolor='none')
        legend_labels.append(input)       

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(0,365)
    ymax = 1.0
    ymin = 0.0

    # load the sensiitvity indices file for a given reservoir release policy
    SI_filedir = 'daily/Soln_' + str(soln_num) + '/release_decisions/' + output_name + '.txt'
    SI = np.loadtxt(SI_filedir, skiprows = 1,delimiter = ",")   # has shape (365000, num_inputs)

    # shift the SI by the min value in each column to make all values positive
    SI_shifted = SI + np.abs(np.min(SI, axis=0))

    SI_avg = np.zeros([365, SI_shifted.shape[1]], dtype=float)
    for i in range(SI_shifted.shape[1]):
        SI_avg[:, i] = averageSensitivity(SI_shifted[:,i], num_years) 
    # shift back to original values
    SI_avg = SI_avg - np.abs(np.min(SI, axis=0))
    
    SI_pos_indices = np.array([])
    SI_neg_indices = np.array([])

    #SI_avg_norm = np.zeros([365, SI_avg.shape[1]], dtype=float)
    y_base = np.zeros([365,], dtype=float)

    # find and plot positive sensitivities
    for i in range(len(input_list)-1):   # get positive first 1st-order interactions
        # find the days where the portion of variance is positive
        y_pos = np.zeros([365], dtype=float)
        SI_pos_indices = np.where(np.sum(SI_avg[:,:],1) > 0)

        # plot the positive first-order portion of variance
        y_pos[SI_pos_indices] = np.sum(SI_avg[:,0:i+1],1)[SI_pos_indices]/np.sum(SI_avg[:,:],1)[SI_pos_indices]
        
        # find current max y-axis value
        ymax = max(ymax, np.max(y_pos))
        ax.fill_between(x, y_base, y_pos, where=y_pos>y_base, color = colors[i])
        y_base = y_pos # on the next iteration, fill between the previous y_pos and the next y_pos

    # find and plot zero or negative sensitivities
    y_neg = np.ones([365,], dtype=float)
    SI_zero_indices = np.where(y_base == 0)
    y_neg[SI_zero_indices] = 0

    # assume all negative sensitivities are caused by interactions
    SI_neg_indices = np.where(np.sum(SI_avg[:, len(input_list)-1::], 1) < 0)
    y_neg[SI_neg_indices] = np.sum(SI_avg[:, len(input_list)-1::], 1)[SI_neg_indices]/np.sum(SI_avg[:,:], 1)[SI_neg_indices]

    ax.fill_between(x, y_base, y_neg, where=y_base<y_neg, color = colors[-1])
    ax.fill_between(x, y_neg, 0, where=y_base>y_neg, color = colors[-1])

    # find current max and min y-axis values
    ymax = max(ymax, np.max(y_neg))
    ymin = min(ymin, np.min(y_neg))
    
    ax.set_ylim(ymin,ymax)
    ax.set_xticks([15,45,75,106,137,167,198,229,259,289,319,350])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=14)

    plt.title(title,fontsize=18)
    
    plt.figlegend(legend_labels,\
                input_list,\
                loc='lower center', ncol=9, fontsize=16, frameon=True)
    
    
    filename = 'daily/Soln_' + str(soln_num) + '/Figures/' + output_name + '_SI_contributions.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
