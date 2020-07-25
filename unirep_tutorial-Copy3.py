
# coding: utf-8

# ## How to use the UniRep mLSTM "babbler". This version demonstrates the 64-unit and the 1900-unit architecture. 
# 
# We recommend getting started with the 64-unit architecture as it is easier and faster to run, but has the same interface as the 1900-unit one.

# Use the 64-unit or the 1900-unit model?

# In[1]:


USE_FULL_1900_DIM_MODEL = False # if True use 1900 dimensional model, else use 64 dimensional one.


# ## Setup

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

if USE_FULL_1900_DIM_MODEL:
    # Sync relevant weight files
    get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/ 1900_weights/')
    
    # Import the mLSTM babbler model
    from unirep import babbler1900 as babbler
    
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./1900_weights"
    
else:
    # Sync relevant weight files
    get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/ 64_weights/')
    
    # Import the mLSTM babbler model
    from unirep import babbler64 as babbler
    
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./64_weights"


# ## Data formatting and management

# Initialize UniRep, also referred to as the "babbler" in our code. You need to provide the batch size you will use and the path to the weight directory.

# In[3]:


batch_size = 50
b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


# UniRep needs to receive data in the correct format, a (batch_size, max_seq_len) matrix with integer values, where the integers correspond to an amino acid label at that position, and the end of the sequence is padded with 0s until the max sequence length to form a non-ragged rectangular matrix. We provide a formatting function to translate a string of amino acids into a list of integers with the correct codex:

# In[4]:


seq_open = open("./emi_iso_seqs.txt")
seq = seq_open.read()


# In[5]:


seq_formatted = []
for line in seq.splitlines():
    seq_formatted.append(np.array(b.format_seq(line)))


# We also provide a function that will check your amino acid sequences don't contain any characters which will break the UniRep model.

# In[ ]:


b.is_valid_seq(seq)


# You could use your own data flow as long as you ensure that the data format is obeyed. Alternatively, you can use the data flow we've implemented for UniRep training, which happens in the tensorflow graph. It reads from a file of integer sequences, shuffles them around, collects them into groups of similar length (to minimize padding waste) and pads them to the max_length. Here's how to do that:

# First, sequences need to be saved in the correct format. Suppose we have a new-line seperated file of amino acid sequences, `seqs.txt`, and we want to format them. Note that training is currently only publicly supported for amino acid sequences less than 275 amino acids as gradient updates for sequences longer than that start to get unwieldy. If you want to train on sequences longer than this, please reach out to us. 
# 
# Sequence formatting can be done as follows:

# In[4]:


# Before you can train your model, 
sequences = []
with open("emi_pos_seqs_7NotY_4.txt", "r") as source:
    with open("formatted.txt", "w") as destination:
        for i,seq in enumerate(source):
            seq = seq.strip()
            sequences.append(seq)
            if b.is_valid_seq(seq) and len(seq) < 275: 
                formatted = ",".join(map(str,b.format_seq(seq)))
                destination.write(formatted)
                destination.write('\n')


# This is what the integer format looks like

# In[ ]:


get_ipython().system('head -n1 formatted.txt')


# Notice that by default format_seq does not include the stop symbol (25) at the end of the sequence. This is the correct behavior if you are trying to train a top model, but not if you are training UniRep representations.

# Now we can use a custom function to bucket, batch and pad sequences from `formatted.txt` (which has the correct integer codex after calling `babbler.format_seq()`). The bucketing occurs in the graph. 
# 
# What is bucketing? Specify a lower and upper bound, and interval. All sequences less than lower or greater than upper will be batched together. The interval defines the "sides" of buckets between these bounds. Don't pick a small interval for a small dataset because the function will just repeat a sequence if there are not enough to
# fill a batch. All batches are the size you passed when initializing the babbler.
# 
# This is also doing a few other things:
# - Shuffling the sequences by randomly sampling from a 10000 sequence buffer
# - Automatically padding the sequences with zeros so the returned batch is a perfect rectangle
# - Automatically repeating the dataset

# In[7]:


bucket_op = b.bucket_batch_pad("formatted.txt", interval=1000) # Large interval


# Inconveniently, this does not make it easy for a value to be associated with each sequence and not lost during shuffling. You can get around this by just prepending every integer sequence with the sequence label (eg, every sequence would be saved to the file as "{brightness value}, 24, 1, 5,..." and then you could just index out the first column after calling the `bucket_op`. Please reach out if you have questions on how to do this.

# Now that we have the `bucket_op`, we can simply `sess.run()` it to get a correctly formatted batch

# In[8]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch = sess.run(bucket_op)
    
print(batch)
print(batch.shape)


# You can look back and see that the batch_size we passed to __init__ is indeed 12, and the second dimension must be the longest sequence included in this batch. Now we have the data flow setup (note that as long as your batch looks like this, you don't need my flow), so we can proceed to implementing the graph. The module returns all the operations needed to feed in sequence and get out trainable representations.

# ## Training a top model and a top model + mLSTM.

# First, obtain all of the ops needed to output a representation

# In[9]:


final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
    b.get_rep_ops())


# `final_hidden` should be a batch_size x rep_dim matrix.
# 
# Lets say we want to train a basic feed-forward network as the top model, doing regression with MSE loss, and the Adam optimizer. We can do that by:
# 
# 1.  Defining a loss function.
# 
# 2.  Defining an optimizer that's only optimizing variables in the top model.
# 
# 3.  Minimizing the loss inside of a TensorFlow session

# In[10]:


y_placeholder = tf.placeholder(tf.float32, shape=[None,1], name="y")
initializer = tf.contrib.layers.xavier_initializer(uniform=False)

with tf.variable_scope("top"):
    prediction = tf.contrib.layers.fully_connected(
        final_hidden, 1, activation_fn=None, 
        weights_initializer=initializer,
        biases_initializer=tf.zeros_initializer()
    )

loss = tf.losses.mean_squared_error(y_placeholder, prediction)


# You can specifically train the top model first by isolating variables of the "top" scope, and forcing the optimizer to only optimize these.

# In[11]:


learning_rate=.005
top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
optimizer = tf.train.AdamOptimizer(learning_rate)
top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
all_step_op = optimizer.minimize(loss)


# We next need to define a function that allows us to calculate the length each sequence in the batch so that we know what index to use to obtain the right "final" hidden state

# In[12]:


def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths

nonpad_len(batch)


# We are ready to train. As an illustration, let's learn to predict the number 42 just optimizing the top model.

# In[ ]:


"""
y = [[1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [1], [0], [1], [1], [1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1], [0], [1], [0], [1], [0], [1], [1], [1], [1], [0], [1], [1], [1], [1], [0], [0], [1], [1], [1], [1], [1], [0], [1], [1], [1], [1], [1], [1], [1], [0], [1], [1]]
num_iters = 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, top_only_step_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: y,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state
                }
        )
        viscosity_pred, final_hidden_layer = sess.run([prediction, final_hidden],
                feed_dict={
                     x_placeholder: batch,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state})
        print("Iteration {0}: {1}".format(i,loss_))
#    print(np.hstack([viscosity_pred, y])
    print(final_hidden_layer)
    plt.scatter(viscosity_pred, y)
    plt.show()
"""


# We can also jointly train the top model and the mLSTM. Note that if using the 1900-unit (full) model, you will need a GPU with at least 16GB RAM. To see a demonstration of joint training with fewer computational resources, please run this notebook using the 64-unit model.

# In[ ]:


"""y = [[1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [1], [0], [1], [1], [1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1], [0], [1], [0], [1], [0], [1], [1], [1], [1], [0], [1], [1], [1], [1], [0], [0], [1], [1], [1], [1], [1], [0], [1], [1], [1], [1], [1], [1], [1], [0], [1], [1]]
num_iters = 25
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, all_step_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: y,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state
                }
        )
        viscosity_pred = sess.run(prediction,
                feed_dict={
                     x_placeholder: batch,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state})
        print("Iteration {0}: {1}".format(i,loss_))
    print(np.hstack([viscosity_pred, y]))
"""


# In[5]:


## 
average_hidden_list = []
final_hidden_list = []
hs_list = []
final_cell_list = []


num2 = range(0, 50)
x = 0
y = 50
for i in num2:
    num1 = range(x, y)
    for j in num1:
        avg_hidden, final_hidden, final_cell, hs_out = (b.get_rep_hs(sequences[j]))
        average_hidden_list.append(avg_hidden)
        final_hidden_list.append(final_hidden)
        final_cell_list.append(final_cell)
        hs_list.append(hs_out)
        print('rep')
    x = x + 50
    y = y + 50
    


# In[ ]:





# In[6]:


average_hidden_pd = pd.DataFrame(np.row_stack(average_hidden_list))
final_hidden_pd = pd.DataFrame(np.row_stack(final_hidden_list))
hidden_state = pd.DataFrame(np.row_stack(hs_list))
print(hidden_state)


# In[7]:


average_hidden_pd.to_csv("emi_pos_reps_7NotY_4.csv")
final_hidden_pd.to_csv("emi_pos_finalhidden_7NotY_4.csv")


# In[4]:


avg_hidden, final_hidden, final_cell = (b.get_rep(QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVRQAPGQGLEWMGRVNPNRRGTTYNQKFEGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARANWLDYWGQGTTVTVSS))


# In[9]:


import pickle
save_loc = "C:\\Users\\pkinn\\Documents\\UniRep\\full representations\\emi larger set\\"
data_name = 'emi_pos_reps_7NotY_4'
file_append = '.pickle'


fn = save_loc + data_name + 'avg_hidden' + file_append
with open(fn, 'wb') as f:
    pickle.dump(average_hidden_list, f)

fn = save_loc + data_name + 'final_hidden' + file_append
with open(fn, 'wb') as f:
    pickle.dump(final_hidden_list, f)

fn = save_loc + data_name + 'final_cell' + file_append
with open(fn, 'wb') as f:
    pickle.dump(final_cell_list, f)
   
fn = save_loc + data_name + 'hidden_state' + file_append
with open(fn, 'wb') as f:
    pickle.dump(hs_list, f)
   
fn = save_loc + data_name + 'all_output_hs' + file_append
with open(fn, 'wb') as f:
    pickle.dump([average_hidden_list, final_hidden_list, final_cell_list, hs_list], f)


# In[ ]:



