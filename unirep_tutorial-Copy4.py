
# coding: utf-8

# In[1]:


USE_FULL_1900_DIM_MODEL = False # if True use 1900 dimensional model, else use 64 dimensional one.


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


# In[3]:


batch_size = 50
b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


# In[ ]:


# Before you can train your model, 
sequences = []
with open("emi_pos_seqs_5G6A_4.txt", "r") as source:
    with open("formatted.txt", "w") as destination:
        for i,seq in enumerate(source):
            seq = seq.strip()
            sequences.append(seq)
            if b.is_valid_seq(seq) and len(seq) < 275: 
                formatted = ",".join(map(str,b.format_seq(seq)))
                destination.write(formatted)
                destination.write('\n')


# In[ ]:


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
    


# In[6]:


average_hidden_pd = pd.DataFrame(np.row_stack(average_hidden_list))
final_hidden_pd = pd.DataFrame(np.row_stack(final_hidden_list))
hidden_state = pd.DataFrame(np.row_stack(hs_list))
print(hidden_state)


# In[7]:


average_hidden_pd.to_csv("emi_pos_reps_5G6A_4.csv")
final_hidden_pd.to_csv("emi_pos_finalhidden_5G6A_4.csv")


# In[8]:


import pickle
save_loc = "C:\\Users\\pkinn\\Documents\\UniRep\\full representations\\emi larger set\\"
data_name = 'emi_pos_reps_5G6A_4'
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




