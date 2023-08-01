# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:45:03 2023

@author: Rohini
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

sns.set_style("dark")

def averageSensitivity(input1, num_years):
    input1_reshaped = np.reshape(input1, (num_years,365))
    avg_1 = np.mean(input1_reshaped, axis = 0)

    return np.transpose(avg_1)

def plot_tvsa_policy(num_years, soln_num, input_list, colors_list, output_name, title):
    colors = colors_list

    legend_labels = []

    for l in range(len(input_list)):
        input = plt.Rectangle((0,0), 1, 1, fc=colors[l], edgecolor='none')
        legend_labels.append(input)       

    SI_filedir = './daily/Soln_' + str(soln_num) + '/release_decisions/' + output_name + '.txt'
    SI = np.loadtxt(SI_filedir, skiprows = 1,delimiter = ",")
    SI_pos = np.array([])
    SI_neg = np.array([])
    for i in range(SI.shape[0]):
        SI_pos = np.where(SI[i,:] >= 0)[0]
        SI_neg = np.where(SI[i,:] < 0)[0]

    #print('SI_pos_cols:', SI_pos)
    #print('SI_neg_cols:', SI_neg)
    
    avg_SI_pos = np.zeros([365, len(SI_pos)], dtype=float)
    avg_SI_neg = np.zeros([365, len(SI_neg)], dtype=float)

    for si_pos in range(len(SI_pos)):
        #print(SI_pos[si_pos])
        avg_SI_pos[:, si_pos] = averageSensitivity(SI[:, SI_pos[si_pos]], num_years)
    for si_neg in range(len(SI_neg)):
        avg_SI_neg[:, si_neg] = averageSensitivity(SI[:, SI_neg[si_neg]], num_years)
    #avg_SI = np.transpose(avg_SI)
    #yr = 4 # 4 for 1936, 59 for 1991

    fig = plt.figure(figsize=(16,8))
    #ymaxs = np.ones(1)
    #ymins = np.zeros(1)

    #y1 = np.zeros([365]) # Days in a year
    ax = fig.add_subplot(1,1,1)
    x = np.arange(0,365)
    avg_SI_pos_norm = np.zeros([avg_SI_pos.shape[0], avg_SI_pos.shape[1]])
    avg_SI_neg_norm = np.zeros([avg_SI_neg.shape[0], avg_SI_neg.shape[1]])
    
    for si_pos in range(len(SI_pos)):
        avg_SI_pos_norm[:, si_pos] = avg_SI_pos[:, si_pos]/np.sum(avg_SI_pos,1) 
    for si_neg in range(len(SI_neg)):
        avg_SI_neg_norm[:, si_neg] = (avg_SI_neg[:, si_neg] + np.min(avg_SI_neg[:, si_neg])/np.sum(avg_SI_neg+np.min(avg_SI_neg),1))*-1

    bottom_pos = np.zeros([365,])
    top_neg = np.zeros([365,])

    for si_pos in range(len(SI_pos)):
        if SI_pos[si_pos] >= 5:
            ax.bar(x, avg_SI_pos_norm[:, si_pos], bottom = bottom_pos, color = colors[-1], width = 1, edgecolor = "none", align='edge')
            bottom_pos = bottom_pos + avg_SI_pos_norm[:, si_pos]
        else:
            ax.bar(x, avg_SI_pos_norm[:, si_pos], bottom = bottom_pos, color = colors[SI_pos[si_pos]], width = 1, edgecolor = "none", align='edge')
            bottom_pos = bottom_pos + avg_SI_pos_norm[:, si_pos]
        
    for si_neg in range(len(SI_neg)):
        if SI_neg[si_neg] >= 5:
            #print(avg_SI_neg_norm[:, si_neg])
            ax.bar(x, top_neg, bottom = avg_SI_neg_norm[:, si_neg], color = colors[-1], width = 1, edgecolor = "none", align='edge')
            top_neg = avg_SI_neg_norm[:, si_neg] + top_neg
        else:
            #print(avg_SI_neg_norm[:, si_neg])
            ax.bar(x, top_neg, bottom = avg_SI_neg_norm[:, si_neg], color = colors[SI_neg[si_neg]], width = 1, edgecolor = "none", align='edge')
            top_neg = avg_SI_neg_norm[:, si_neg] + top_neg
    

    ax.set_xlim(ax.patches[0].get_x(), ax.patches[-1].get_x())
    ax.set_ylim(-2,2)
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
