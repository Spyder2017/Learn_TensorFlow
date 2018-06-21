def attention(inputs, attention_size, time_major=False):  
	if isinstance(inputs, tuple):  
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.  
		inputs = tf.concat(inputs, 2)  
  
	if time_major:  
		# (T,B,D) => (B,T,D)  
		inputs = tf.transpose(inputs, [1, 0, 2])  
  
	inputs_shape = inputs.shape  
	sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer  
	hidden_size = inputs_shape[2].value  # hidden size of the RNN layer  
  
	# Attention mechanism  
	W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  
	b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
	u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
  
	v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))  
	vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))  
	exps = tf.reshape(tf.exp(vu), [-1, sequence_length])  
	alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  
  
	# Output of Bi-RNN is reduced with attention vector  
	output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)  
  
	return output  
  
  
# cnn的输出为B*4*8*128  
chunk_size = 128	
chunk_n = 32	
rnn_size = 256	
attention_size = 50  
n_output_layer = MAX_CAPTCHA*CHAR_SET_LEN   # 输出层	
  
# 定义待训练的神经网络	
def recurrent_neural_network():
	data = crack_captcha_cnn()
	 
	data = tf.reshape(data, [-1, chunk_n, chunk_size])	
	data = tf.transpose(data, [1,0,2])	
	data = tf.reshape(data, [-1, chunk_size])	
	data = tf.split(data,chunk_n)  
	  
	# 只用RNN  
	#layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}   
	#lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)   
	#outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)	
	#ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])	
	  
	# RNN + Attention	  
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)   
	outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)	   
	attention_output = attention(outputs, attention_size, True)  
	  
	# output  
	drop = tf.nn.dropout(attention_output, keep_prob)  
	# Fully connected layer  
	W = tf.Variable(tf.truncated_normal([rnn_size, n_output_layer], stddev=0.1), name="W")  
	b = tf.Variable(tf.constant(0., shape=[n_output_layer]), name="b")  
	  
	ouput = tf.nn.xw_plus_b(drop, W, b, name="scores")  
	return ouput  