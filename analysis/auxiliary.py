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
from datasets import Dataset
from transformers import DefaultDataCollator
from zlib import crc32
from sklearn.metrics import ndcg_score
import tensorflow as tf


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


def process_pandas_to_tfdataset(df, tokenizer, max_length=80, shuffle=True, text_col='text', target_col='label', batch_size=8):
    """
    Prepare NLP data in a Pandas DataFrame to be used 
    in a TensorFlow transformer model.
    
    Parameters
    ----------
    df : DataFrame
        The corpus, containing the columns `text_col` 
        (the sentences) and `target_col` (the labels).
    tokenizer : HuggingFace AutoTokenizer
        A tokenizer loaded from 
        `transformers.AutoTokenizer.from_pretrained()`.
    max_length : int
        Maximum length of the sentences (smaller 
        sentences will be padded and longer ones
        will be truncated). This is required for 
        training, so batches have instances of the
        same shape.
    shuffle : bool
        Shuffle the dataset order when loading. 
        Recommended True for training, False for 
        validation/evaluation.
    text_col : str
        Name of `df` column containing the sentences.
    target_col : str
        Name of `df` column containing the labels of 
        the sentences.
    batch_size : int
        The size of the batch in the output 
        tensorflow dataset.
        
    Returns
    -------
    tf_dataset : TF dataset
        A dataset that can be fed into a transformer 
        model.
    """
    
    # Security checks:
    renamed_df = df.rename({target_col:'labels'}, axis=1) # Hugging Face requer esse nome p/ y.
    
    # Define função para processar os dados com o tokenizador:
    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding=True, max_length=max_length, truncation=True)
    
    # pandas -> hugging face:
    hugging_set = Dataset.from_pandas(renamed_df)
    # texto -> sequência de IDs: 
    encoded_set = hugging_set.map(tokenize_function, batched=True)
    
    # hugging face -> tensorflow dataset:
    data_collator = DefaultDataCollator(return_tensors="tf")
    tf_dataset = encoded_set.to_tf_dataset(columns=["attention_mask", "input_ids", "token_type_ids"], label_cols=["labels"], shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)
    
    return tf_dataset


def count_tokens(string, tokenizer, max_length=None):
    """
    Count the number of tokens (int) in `string` (str) returned by 
    `tokenizer` (BertTokenizer), clipping it to `max_length` (int or 
    None).
    """

    if max_length is None:
        truncation = False
    else:
        truncation = True

    n_tokens = len(tokenizer(string, padding=True, max_length=max_length, truncation=truncation)['input_ids'])
    return n_tokens


def crop_to_max_tokens(string, tokenizer, max_length=512):
    """
    Crop string to a maximum number of tokens.
    """
    tokens = tokenizer.tokenize(string)[:max_length]
    cropped = tokenizer.convert_tokens_to_string(tokens)
    
    return cropped


def skip_preamble(text_series, preamble_regex):
    """
    Returns the substrings that follow the regular expression.
    
    Parameters
    ----------
    text_series : Series
        Series containing strings to be cropped.
    preamble_regex : str
        Regular expression representing the start of the important
        segment of the text.
    
    Returns
    -------
    final_text : Series
        Texts that follow `preamble_regex`. If the regex is not 
        found, return the input text.
    """
    
    # Select text that follows the regex:
    sel_text = text_series.str.extract('(?:' + preamble_regex + ')(.*)')[0]
    # Use the whole text if regex is not found:
    final_text = sel_text.fillna(text_series)
    
    return final_text


def to_titlecase(match):
    """
    Transform the string in a regex `match` group to title case.
    """
    return match.group(0).title()


def all_caps_to_title(text_series, min_len=3):
    """
    Transform all caps substrings in entries in the `text_series` 
    (Series) with length `min_len` (int) or greater to title case.
    """
    return text_series.str.replace('([A-ZÇÃÁÀÂÊÉÍÓÕÔÚ]{' + str(min_len) + ',})', to_titlecase, regex=True)


def limit_ellipsis(text_series, max_len=3):
    """
    Limit the size of a sequence of periods in the strings in 
    Series `text_series` to `max_len` (int).
    """
    
    # Create replacement:
    ellipsis = ''.join(['.'] * max_len)
    # Set search regex:
    regex = r'\.{' + str(max_len) + ',}'
    
    return text_series.str.replace(regex, ellipsis, regex=True)


def hash_string(string, prefix=''):
    """
    Takes a `string` as input, remove `prefix` from it and turns it into a hash.
    """
    name   = string.replace(prefix, '')
    return crc32(bytes(name, 'utf-8'))


def test_set_check_by_string(string, test_frac, prefix=''):
    """
    Returns a boolean array saying if the data identified by `string` belongs to the test set or not.
    
    Parameters
    ----------
    string : str
        The string that uniquely identifies an example.
    test_frac : float
        The fraction of the complete dataset that should go to the test set (0 to 1).
    prefix : str (default '')
        A substring to remove from `string` before deciding where to place the example.
        
    Returns
    -------
    is_test : bool
        A bool number saying if the example belongs to the test set.
    """

    return hash_string(string, prefix) & 0xffffffff < test_frac * 2**32


def train_test_split_by_string(df, test_frac, col, prefix=''):
    """
    Split a DataFrame `df` into train and test sets based on string hashing.
    
    Input
    -----
    
    df : Pandas DataFrame
        The data to split.
        
    test_frac : float
        The fraction of `df` that should go to the test set (0 to 1).

    col : str or int
        The name of the `df` column to use as identifier (to be hashed).
        
    prefix : str (default '')
        A substring to remove from the rows in column `col` of `df` 
        before deciding where to place the example.
        
    Returns
    -------
    
    The train and the test sets (Pandas DataFrames).
    """
    ids = df[col]
    in_test_set = ids.apply(lambda s: test_set_check_by_string(s, test_frac, prefix))
    return df.loc[~in_test_set], df.loc[in_test_set]


def tf_ndcg(y_true, y_pred, k=None):
    """
    Normalized Discounted Cumulative Gain (NDCG) metric implementation in TensorFlow.
    Args:
        y_true: Tensor of shape (batch_size, num_items) with the true relevance scores.
        y_pred: Tensor of shape (batch_size, num_items) with the predicted relevance scores.
        k: The maximum number of items to consider in the ranking.
    Returns:
        NDCG metric value as a scalar tensor.
    """

    # Bug fix:
    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)
    
    # Use all data if k is not provided:
    if k == None:
        k = tf.shape(y_pred)[1]
    
    # Get the indices that order the predictions (descending):
    _, indices = tf.math.top_k(y_pred, k=k)
    # Order the true labels according to the predictions:
    y_true = tf.gather(y_true, indices, batch_dims=1)
    # Compute the gain of each treu label:
    gain = tf.pow(2.0, y_true) - 1.0
    
    discounts = tf.math.log1p(tf.cast(tf.range(1, k+1), dtype=tf.float32)) / tf.math.log(2.0)
    dcg = tf.reduce_sum(gain / discounts, axis=1)
    idcg = tf.reduce_sum(tf.sort(gain, axis=1, direction='DESCENDING')[:, :k] / discounts, axis=1)
    ndcg = dcg / idcg
    return tf.reduce_mean(ndcg)


def my_ndcg(y_true, y_pred):
    """
    Compute the Normalized Discounted Cumulative Gain (nDCG)
    metric.
    
    Parameters
    ----------
    y_true : 1D array
        True relevance, from the lowest (0) to the highest.
    y_pred : 1D array
        Predicted relevance, used for sorting the instances.
    
    Returns
    -------
    nDCG : float
        The metric.
    """
    # Order true labels based on predictions:
    sorted_idx  = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_idx]
    
    # Compute gain from relevance:
    exp_gain = 2**sorted_true - 1
    # Compute order discount:
    discount = np.log2(np.arange(1, 1 + len(y_true)) + 1)
     
    # Compute the Discounted Cumulative Gain: 
    DCG = np.sum(exp_gain / discount)  
    # Compute the same for the perfect ordering:
    ideal_gain = exp_gain[np.argsort(exp_gain)[::-1]]
    iDCG = np.sum(ideal_gain / discount)  
    
    # Normalize:
    nDCG = DCG / iDCG
    
    return nDCG


def ndcg_metric(y_true, y_pred):
    """
    The Normalized Discounted Cumulative Gain (nDCG) metric
    with input shape the same as other regression metrics.
    
    Parameter
    ---------
    y_true : 1D array of floats or ints
        True relevance of the instances. For a 5-level 
        scale, they should be {0, 1, 2, 3, 4}.
    y_true : 1D array of floats
        Predicted relevance of the instances.
    
    Returns
    -------
    
    ndcg : float
        The nDCG score.
    """
    return ndcg_score(np.array([y_true]), np.array([y_pred]))
