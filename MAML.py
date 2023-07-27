"""
Filename: KBHB_site_pred_MAML.py
Author: yellower
"""
from torch.utils.data import DataLoader
import pandas as pd
from keras.models import load_model
import numpy as np
from keras import backend as K
from tqdm import tqdm
from utils import to_categorical,caculate_auc
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers


from keras.models import Model
from keras.layers import *

import tensorflow as tf
import numpy as np



'''多头Attention'''


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


'''Transformer的Encoder部分'''


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.05):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-12)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


'''Transformer输入的编码层'''


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions









class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        lr_inner = 0.01
        # print(len(data))
        [support_x,query_x],[support_y,query_y] = data
        print(support_x)
        # print(Y.shape)
        with tf.GradientTape() as test_tape:
            with tf.GradientTape() as train_tape:
                # predictions = model(tf.convert_to_tensor(support_x), training=True)
                predictions = model(support_x, training=True)
                loss = self.compiled_loss(support_y, predictions)
            gradients = train_tape.gradient(loss, model.trainable_variables)

            k = 0
            # model_weights = self.weights()
            model_copy = Model(inputs=inputs, outputs=outputs)
            model_copy.set_weights(self.get_weights())
            for j in range(len(model_copy.layers)):
                # model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                #                                           tf.multiply(lr_inner, gradients[k]))
                # model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                #                                         tf.multiply(lr_inner, gradients[k + 1]))
                for weight_index in range(len(model_copy.layers[j].weights)):
                    model_copy.layers[j].weights[weight_index] = tf.subtract(model.layers[j].weights[weight_index],
                                                                             tf.multiply(lr_inner, gradients[k]))
                    k += 1
            # optimizer_copy.apply_gradients(zip(gradients, model_copy.trainable_variables))
            predictions_query = model_copy(tf.convert_to_tensor(query_x), training=True)
            test_loss = self.compiled_loss(predictions_query, query_y)

        # Step 8
        test_gradients = test_tape.gradient(test_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(test_gradients, self.trainable_variables))
        self.compiled_metrics.update_state(query_y,predictions_query)
        return {m.name: m.result() for m in self.metrics}






def loss_function(pred_y, y):
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits
def compute_gradients(model, x, y, loss_fn=loss_function):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss
def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


class SineModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.inputs = layers.Input(shape=(maxlen,))
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.hidden1 = keras.layers.Dense(40, input_shape=(1,))
        self.hidden2 = keras.layers.Dense(40)
        self.out = keras.layers.Dense(1)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        O_seq = Dropout(0.05)(x)
        O_seq = Dense(32, activation='selu')(O_seq)
        O_seq = Dropout(0.05)(O_seq)
        outputs = Dense(2, activation='softmax')(O_seq)
        return outputs


def copy_model(model, x):
    '''Copy model weights to a new model.

    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = SineModel()

    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))

    copied_model.set_weights(model.get_weights())
    return copied_model


def train_maml(num_folds,model, epochs, dataset, fasta_K_emb_arrays, labels_arrays,query_size,lr_inner=0.01):
    '''Train using the MAML setup.

    The comments in this function that start with:

        Step X:

    Refer to a step described in the Algorithm 1 of the paper.

    Args:
        model: A model.
        epochs: Number of epochs used for training.
        dataset: A dataset used for training.

        batch_size: Batch size. Default value is 1. The paper does not specify
            which value they use.
       fasta_K_emb_arrays: fasta_K_sequences
       labels_arrays: labels for fasta_K
       query_size: query set datasets.
       lr_inner: Inner learning rate (alpha in Algorithm 1). Default value is 0.01.

    Returns:
        A strong, fully-developed and trained maml.
    '''

    skf = StratifiedKFold(n_splits = num_folds)
    logits_test_concat = np.array([])
    labels_test_concat = np.array([])
    loss_object = tf.keras.losses.BinaryCrossentropy()

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    for fold, (train_idx, val_idx) in enumerate(skf.split(fasta_K_emb_arrays, labels_arrays)):
        fasta_K_emb_train,labels_train,fasta_K_emb_test,labels_test = fasta_K_emb_arrays[train_idx], labels_arrays[train_idx],fasta_K_emb_arrays[val_idx],labels_arrays[val_idx]
        train_acc_metric = keras.metrics.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.Adam()
        for epoch in tqdm(range(epochs)):
            for K in range(opt.K_tasks):
                label_train_pos_index = np.where(labels_train == 1)[0]
                label_train_neg_index = np.random.choice(np.where(labels_train == 0)[0],len(label_train_pos_index))
                fasta_K_emb_train_pos = fasta_K_emb_train[label_train_pos_index]
                fasta_K_emb_train_neg = fasta_K_emb_train[label_train_neg_index]
                label_train_pos = labels_train[label_train_pos_index]
                label_train_neg = labels_train[label_train_neg_index]
                fasta_K_emb_train_task = np.append(fasta_K_emb_train_pos,fasta_K_emb_train_neg,axis=0)
                labels_train_task = to_categorical(np.append(label_train_pos,label_train_neg,axis=0),2).astype('float')
                train_dataset = tf.data.Dataset.from_tensor_slices((fasta_K_emb_train_task, labels_train_task))
                train_dataset = train_dataset.batch(opt.batch_size)
                for idx, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    dataset = train_test_split(x_batch_train.numpy(), y_batch_train.numpy(), test_size=query_size, stratify=y_batch_train.numpy())

                    support_x, query_x, support_y, query_y = dataset

                    with tf.GradientTape() as test_tape:
                        with tf.GradientTape() as train_tape:
                            predictions = model(tf.convert_to_tensor(support_x), training=True)
                            loss = loss_object(support_y,predictions)
                        gradients = train_tape.gradient(loss, model.trainable_weights)
                        k = 0
                        model_weights = model.get_weights()
                        # model_copy = Model(inputs=inputs, outputs=outputs)
                        model_copy = Model(inputs,outputs)
                        # model_copy.forward(tf.convert_to_tensor(support_x))
                        model_copy.set_weights(model_weights)
                        for j in range(len(model_copy.layers)):
                            # model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                            #                                           tf.multiply(lr_inner ,gradients[k]))
                            # model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                            #                                         tf.multiply(lr_inner, gradients[k + 1]))
                            for weight_index in range(len(model_copy.layers[j].weights)):
                                model_copy.layers[j].weights[weight_index] = tf.subtract(model.layers[j].weights[weight_index],tf.multiply(lr_inner, gradients[k]))
                                k += 1
                        # optimizer_copy.apply_gradients(zip(gradients, model_copy.trainable_variables))
                        predictions_query = model_copy(tf.convert_to_tensor(query_x), training=True)
                        test_loss = loss_object(query_y,predictions_query)
                        # Step 8
                    test_gradients = test_tape.gradient(test_loss,model_copy.trainable_weights)
                    optimizer.apply_gradients(zip(test_gradients, model.trainable_weights))
                    # Update training metric.
                    train_acc_metric.update_state(query_y,predictions_query)
                #Display metrics at the end of each epoch.
                train_acc = train_acc_metric.result()
                print("Training acc over epoch: %.4f" % (float(train_acc),))
                # Reset training metrics at the end of each epoch
                train_acc_metric.reset_states()

        logits_test = model.predict(fasta_K_emb_test)

        fpr, tpr, roc_auc = caculate_auc(labels_test, logits_test[:, 1])
        print(fold)
        print(roc_auc)
        logits_test_concat = np.append(logits_test_concat, logits_test[:, 1])
        labels_test_concat = np.append(labels_test_concat, labels_test)

