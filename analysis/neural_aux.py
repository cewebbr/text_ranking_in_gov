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


from datasets import Dataset
from transformers import DefaultDataCollator
import tensorflow as tf


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
    # Order the true labels according to the predictions and make lowest relevance 1 -> 0:
    y_true = tf.gather(y_true, indices, batch_dims=1) - 1.0
    # Compute the gain of each treu label:
    gain = tf.pow(2.0, y_true) - 1.0
    
    discounts = tf.math.log1p(tf.cast(tf.range(1, k+1), dtype=tf.float32)) / tf.math.log(2.0)
    dcg = tf.reduce_sum(gain / discounts, axis=1)
    idcg = tf.reduce_sum(tf.sort(gain, axis=1, direction='DESCENDING')[:, :k] / discounts, axis=1)
    ndcg = dcg / idcg
    return tf.reduce_mean(ndcg)


class NDCGMetric(tf.keras.metrics.Metric):
    def __init__(self, max_size, init_value=1, name='nDCG', **kwargs):
        """
        Create a Normalized Discounted Cumulative Gain Metric
        for evaluating a model during training.
        """

        super().__init__(name=name, **kwargs)
        self.max_size = max_size
        self.init_value = init_value
        
        self.true_init   = tf.keras.initializers.Constant(init_value)
        self.true_labels = self.add_weight(name='true_labels', shape=(max_size, 1), dtype=tf.float32, initializer=self.true_init)
        self.pred_init   = tf.keras.initializers.Constant(0)
        self.pred_labels = self.add_weight(name='pred_labels', shape=(max_size, 1), dtype=tf.float32, initializer=self.pred_init)
        self.n_stored    = self.add_weight(name='n_stored', dtype=tf.int32, initializer='zeros')
        self.batch_size  = self.add_weight(name='batch_size', dtype=tf.int32, initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        self.batch_size.assign(tf.shape(y_true)[0])
        self.true_labels.assign(tf.concat([self.true_labels[:self.n_stored], y_true, self.true_labels[self.n_stored + self.batch_size:]], axis=0))
        self.pred_labels.assign(tf.concat([self.pred_labels[:self.n_stored], y_pred, self.pred_labels[self.n_stored + self.batch_size:]], axis=0))
        self.n_stored.assign(self.n_stored + self.batch_size)
        
    def result(self):
        return tf_ndcg(self.true_labels[:self.n_stored], self.pred_labels[:self.n_stored])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'max_size':self.max_size, 'init_value':self.init_value}

    def reset_state(self):
        self.true_labels.assign(self.true_init((self.max_size, 1), dtype=tf.float32))
        self.pred_labels.assign(self.pred_init((self.max_size, 1), dtype=tf.float32))
        self.n_stored.assign(0)
        self.batch_size.assign(0)


    
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
