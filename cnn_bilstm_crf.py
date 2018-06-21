class CNN_Bi_LSTM_NER(neural_tagger):

	def __str__(self):
		return "CNN-BiLSTM-CRF NER"

	def build(self):
		with tf.name_scope('weigths'):
			self.W = tf.get_variable(
				shape=[self.hidden_dim * 2, self.nb_classes],
				initializer=tf.truncated_normal_initializer(stddev=0.01),
				name='weights'
			)
			self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
			self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
			# extract bigram feature, so filter's shape :[1, 2*emb_dim, 1,
			# hidden_dim]
			self.conv_weight = tf.get_variable(
				shape=[2, self.emb_dim, 1, self.emb_dim],
				initializer=tf.truncated_normal_initializer(stddev=0.01),
				name='conv_weights'
			)

		with tf.name_scope('biases'):
			self.b = init_variable([self.nb_classes], name="bias")
			self.conv_bias = init_variable([self.hidden_dim], name="conv_bias")
		return

	def inference(self, X, X_len, reuse=None):
		word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
		word_vectors = tf.nn.dropout(word_vectors, keep_prob=self.keep_prob)
		# word_vectors = tf.reshape(
		# word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])

		with tf.variable_scope('convolution'):
			word_vectors = tf.reshape(
				word_vectors, [-1, self.templates, self.emb_dim, 1])
			conv = tf.nn.conv2d(word_vectors, self.conv_weight,
								strides=[1, 1, 1, 1], padding='VALID')
			conv = conv + self.conv_bias
			conv = tf.reshape(
				conv, [-1, self.time_steps, (self.templates - 1) * self.emb_dim])
		word_vectors = tf.reshape(
			word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])
		word_vectors = tf.concat(2, [word_vectors, conv])

		with tf.variable_scope('label_inference', reuse=reuse):
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self.lstm_fw,
				cell_bw=self.lstm_bw,
				inputs=word_vectors,
				dtype=tf.float32,
				sequence_length=X_len
			)
			outputs = tf.concat(2, [outputs[0], outputs[1]])
			outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
			# outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

		with tf.name_scope('linear_transform'):
			scores = tf.matmul(outputs, self.W) + self.b
			# scores = tf.nn.softmax(scores)
			scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
		return scores

	def loss(self, pred):
		with tf.name_scope('loss'):
			log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
				pred, self.Y, self.X_len)
			cost = tf.reduce_mean(-log_likelihood)
			reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.conv_weight)\
				+ tf.nn.l2_loss(self.b) + tf.nn.l2_loss(self.conv_bias)
			if self.fine_tuning:
				reg += tf.nn.l2_loss(self.emb_matrix)
			cost += reg * self.l2_reg
			return cost