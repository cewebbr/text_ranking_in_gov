#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auxiliary code for ICEDEG 2023 tutorial on Overseeing government with AI.
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as pl
import numpy as np

def multicircles(x, y, d=0.035):
    """
    Create a list of circle patches located at each (x,y).
    """
    
    assert len(x) == len(y), 'x and y must have the same length.'
    plist = []
    for xx, yy in zip(x, y):
        plist.append(patches.Ellipse((xx, yy), d, d))
    
    return plist


def mango_scatter_plot(x, y, k):
    """
    Scatter plot for the tastyness classification of mangos.
    
    Parameters
    ----------
    x : array of float
        Color of the mangos.
    y : array of float
        Consistency of the mangos.
    k : array of int
        Tastyness of the mangos.
    """
    
    # Color according to class 'k':
    cmap = pl.get_cmap('tab10')
    colors = np.array([cmap(i) for i in k])
    
    # Scatter plot:
    pl.scatter(x, y, color=colors, alpha=0.3)
    
    # Format:
    pl.xlim([0,1])
    pl.ylim([0,1])
    pl.xlabel('Color')
    pl.ylabel('Consistency')
    
    
def plot_predictions(clf):
    """
    Plot the decision boundary of a binary classifier `clf`
    fitted on two features ranging from 0 to 1.
    
    Copied from Aurelien Geron's Hands-on Machine Learning
    with Scikit-Learn, Keras and Tensorflow, 2nd ed.
    """
    axes = [0., 1., 0., 1.]
    x0s = np.linspace(axes[0], axes[1], 200)
    x1s = np.linspace(axes[2], axes[3], 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    #y_decision = clf.decision_function(X).reshape(x0.shape)
    pl.contour(x0, x1, y_pred, colors='b', linewidths=0.5, alpha=0.4)
    #pl.contourf(x0, x1, y_decision, cmap=pl.cm.brg, alpha=0.1)
    

def biased_random_split(features, target, train_frac, selection_bias, rng):
    """
    Take a biased subsample, in which the probability that
    an instance is selected depends on the target.
    
    Parameters
    ----------
    features : tuple of arrays
        Predictor variables X of the data.
    target : array of ints
        Binary labels ofthe data
    train_frac: float
        The subsample size as a fraction of the dataset.
    selection_bias : float
        How much more likely the positive class is to be
        selected, in percentage.
    rng : Numpy random number generator
        Used to randomly select the data.
    
    Returns
    -------
    sel_features : tuple of arrays
        Each array is one feature of the selected subsample, 
        in the same order as the input.
    sel_target : array
        The targets of the selected subsample.
    """
    
    # Compute the probability of selecting an instance for the training set based on the target (tastyness):
    pos_frac = target.mean()
    neg_sel_prob = train_frac / (1 + pos_frac * selection_bias)
    pos_sel_prob = neg_sel_prob * (1 + selection_bias)
    
    # Print probabilities of selection:
    print('Probability of adding instance to training set, according to its target label:')
    print('label == 1: {:.1f}%;   label == 0: {:.1f}%'.format(pos_sel_prob * 100, neg_sel_prob * 100))
    
    # Define whether each instance was selected:
    instance_sel_prob = np.where(target == 1, pos_sel_prob, neg_sel_prob)
    train_selected = rng.random(len(target)) < instance_sel_prob
    
    # Select instances:
    sel_features = tuple(f[train_selected] for f in features)
    sel_target   = target[train_selected]
    
    return sel_features, sel_target


def print_arrays_heads(arrays, names, n=5):
    """
    Print the name of the arrays and the first `n` entries of 
    each of the input arrays.
    
    Parameters
    ----------
    arrays : iterable of arrays
        Arrays to print.
    names : iterable of str
        Name of the arrays
    n : int
        Number of first entries to print.
    """
    name_len = max([len(name) for name in names])
    
    for arr, name in zip(arrays, names):
        print((name + ':').ljust(name_len + 1), arr[:n])