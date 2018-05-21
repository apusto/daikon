#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import tensorflow as tf

from daikon import constants as C


def compute_lengths(sequences):
    """
    This solution is similar to:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.abs(sequences))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths


def define_computation_graph(source_vocab_size: int, target_vocab_size: int, batch_size: int, model_name: str):
    tf.reset_default_graph()

    # Placeholders for inputs and outputs
    encoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')

    decoder_targets = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')

    keep_prob = tf.placeholder(tf.float32, name='dropout_probability')

    with tf.variable_scope("Embeddings"):
        source_embedding = tf.get_variable('source_embedding', [source_vocab_size, C.EMBEDDING_SIZE])
        target_embedding = tf.get_variable('target_embedding', [source_vocab_size, C.EMBEDDING_SIZE])

        encoder_inputs_embedded = tf.nn.embedding_lookup(source_embedding, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(target_embedding, decoder_inputs)
        print(encoder_inputs_embedded.get_shape())
        print(decoder_inputs_embedded.get_shape())

    with tf.variable_scope("Encoder"):
        encoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        encoder_cell_back = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        initial_state = encoder_cell.zero_state(batch_size, tf.float32)
        initial_state_back = encoder_cell.zero_state(batch_size, tf.float32)

        if 'original' in model_name:
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                     encoder_inputs_embedded,
                                                                     initial_state=initial_state,
                                                                     dtype=tf.float32)
        elif 'bidirectional' in model_name:
            # Each of the RNNs has an output of size HIDDEN_SIZE, plus for each entry in the batch a hidden state. This
            # means the resulting tensors are of shape [BATCH_SIZE, SENTENCE_LENGTH (variable), HIDDEN_SIZE] and
            # tensors for the cell output [BATCH_SIZE, HIDDEN_SIZE] as well as state [BATCH_SIZE, HIDDEN_SIZE].
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(encoder_cell,
                                                                     encoder_cell_back,
                                                                     encoder_inputs_embedded,
                                                                     initial_state_fw=initial_state,
                                                                     initial_state_bw=initial_state_back,
                                                                     dtype=tf.float32)
            # print(outputs[0].get_shape())
            # print(output_states[0].c.get_shape())
            # print(output_states[0].h.get_shape())
            # print(outputs[1].get_shape())
            # print(output_states[1].c.get_shape())
            # print(output_states[1].h.get_shape())
            encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
                tf.concat([output_states[0].c, output_states[1].c], 1),
                tf.concat([output_states[0].h, output_states[1].h], 1))
            # print(encoder_final_state_combined[0].get_shape())
            # print(encoder_final_state_combined[1].get_shape())

    with tf.variable_scope("Decoder"):
        # decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=10)
        if 'original' in model_name:
            decoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        elif 'bidirectional' in model_name:
            decoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE * 2)
        # decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        #     cell=decoder_cell,
        #     embedding=decoder_inputs_embedded,
        #
        # )

        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                 decoder_inputs_embedded,
                                                                 initial_state=encoder_final_state,
                                                                 dtype=tf.float32)

        # print(decoder_outputs.get_shape())
        # print(decoder_final_state.c.get_shape())
        # print(decoder_final_state.h.get_shape())
        drop_out = tf.nn.dropout(decoder_outputs, keep_prob)

    with tf.variable_scope("Logits"):
        decoder_logits = tf.contrib.layers.linear(drop_out, target_vocab_size)
        print("decoder_logits", decoder_logits.get_shape())
        print("decoder_targets", decoder_targets.get_shape())
        unk_supressor = tf.one_hot(tf.scalar_mul(2, tf.ones(tf.shape(decoder_targets), tf.int32)),
                                   depth=target_vocab_size, axis=-1, on_value=0.0, off_value=1.0)
        print("unk_supressor", unk_supressor.get_shape())
        decoder_logits = tf.multiply(decoder_logits, unk_supressor)
        print("decoder_logits after", decoder_logits.get_shape())

    with tf.variable_scope("Loss"):
        one_hot_labels = tf.one_hot(decoder_targets, depth=target_vocab_size, dtype=tf.float32)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=decoder_logits)

        # mask padded positions
        target_lengths = compute_lengths(decoder_targets)
        target_weights = tf.sequence_mask(lengths=target_lengths, maxlen=None, dtype=decoder_logits.dtype)
        weighted_cross_entropy = stepwise_cross_entropy * target_weights
        loss = tf.reduce_mean(weighted_cross_entropy)

    with tf.variable_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, summary, keep_prob
