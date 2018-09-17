# CPC_ActiveInference2018

Active Inference Tutorial from the Computational Psychiatry Conference in Zurich 2018 (http://www.translationalneuromodeling.org/cpcourse/)

This tutorial containts three parts ('Practical_*' I to III) plus some auxiliary SPM-functions that were adjusted for these particular examples (mainly to adjust the figures). Only other requirement to run the code in Matlab is SPM12 and particularly the DEM toolbox of SPM.

Practical_I.m illustrates how to implement a task as active inference, where an agent has to solve the trade-off between gaining information about the world and maximising reward. This information-gain refers to 'hidden state exploration', which allows agents to perform accurate inference about the current hidden state (context). Practical_I_Solutions.m contains some additional tasks that explore the role of different parts of the computational architecture in modelling behaviour of this task, such as an agent's prior preferences, learning rate or precision.

Practical_II.m illustrates model inversion (parameter estimation) based on simulated behaviour in the task that was introduced in Practical_I.m. This includes routines for parameter estimation based on variational Bayes, design optimisation for parameter recovery, inference on group differences and crossvalidation.

Practical_III.m illustrates a second task with a different type of information-gain, namely 'model-parameter exploration', where an agent performs goal-directed exploration to decrease uncertainty about a particular option and thus perform active learning. 

Practical_I.m and Practical_III.m also reproduce some of the simulations discussed here: https://www.biorxiv.org/content/early/2018/09/07/411272, which discusses the distinction between active inference and active learning in more detail. 
