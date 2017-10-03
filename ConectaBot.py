import numpy as np
import tensorflow as tf


def ConectaBot(A, no_jugador):

	MT = np.loadtxt("MatrixS.txt", dtype='i', delimiter = ',')

	MT = MT.transpose()

	MTx = MT[0:41]
	MTy = MT[42]

	MTx = MTx.transpose()

	m, n = MTx.shape

	X = np.c_[np.ones((m, 1)), MTx]

	n_epochs = 5000
	learning_rate = 0.001

	x = tf.constant(X, dtype=tf.float32, name = "x")
	y = tf.constant(MTy.reshape(-1, 1), dtype=tf.float32, name = "y")
	theta = tf.Variable(tf.random_uniform([n + 1, 1], 0, 1.0), name = "theta")
	y_pred = tf.matmul(x, theta, name = "predictions")
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name = "mse")
	gradients = 2/m * tf.matmul(tf.transpose(x), error)
	#training_op = tf.assign(theta, theta - learning_rate * gradients)
	optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9)
	training_op = optimizer.minimize(mse)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(n_epochs):
			#if epoch % 100 == 0:
				#print("Epoch", epoch, "MSE =", mse.eval())
			sess.run(training_op)
		best_theta = theta.eval()

	A1 = A[0]
	A2 = A[1]
	A3 = A[2]
	A4 = A[3]
	A5 = A[4]
	A6 = A[5]

	A_2 = np.concatenate((A1,A2,A3,A4,A5,A6), axis = 0)

	best_theta = np.transpose(best_theta)

	myarray = np.ones(shape = (42,1))


	for tamaño in range (0,42):
		if (A_2[tamaño] != 0):
			myarray[tamaño,0] = A_2[tamaño]


	mya =tf.constant(myarray, dtype=tf.float32, name = "mya")
	bt =tf.constant(best_theta, dtype=tf.float32, name = "bt")

	res = tf.matmul(bt,mya, name = "respuesta")

	with tf.Session() as sess:
		sess.run(init)
		a = res.eval()
	

	if (a < 0.5):
		r = 0
	if (0.5 <= a < 1.5):
		r = 1
	if (1.5 <= a < 2.5):
		r = 2
	if (2.5 <= a < 3.5):
		r = 3
	if (3.5 <= a < 4.5):
		r = 4
	if (4.5 <= a < 5.5):
		r = 5
	if (5.5 <= a < 6.5):
		r = 6
	if (6.5 <= a <= 7):
		r = 7

	print ("La maquina tiro = ", r)

	return r
	


def main():

	width, height = 7, 6

	board = [[0 for y in range(width)] for x in range(height)]

	a = ConectaBot(board, 1)

	print(a)

if __name__ == '__main__':
	main()