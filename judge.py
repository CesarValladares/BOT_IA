import random
import numpy as np
import tensorflow as tf

width, height = 7, 6

# board = [[0 for x in range(width)]for y in range (height)]
board = [[0 for y in range(width)] for x in range(height)]


# Matrix positions
# [5][0] [5][1] [5][2] [5][3] [5][4] [5][5] [5][6]
# [4][0] [4][1] [4][2] [4][3] [4][4] [4][5] [4][6]
# [3][0] [3][1] [3][2] [3][3] [3][4] [3][5] [3][6]
# [2][0] [2][1] [2][2] [2][3] [2][4] [2][5] [2][6]
# [1][0] [1][1] [1][2] [1][3] [1][4] [1][5] [1][6]
# [0][0] [0][1] [0][2] [0][3] [0][4] [0][5] [0][6]

# This is the function used to make a move.
# The function recieves the column (that is going to be given by the intelligent algorithm), and the player number that is making the move.
# If the move can't be made, the function returns -1, else: it returns the col in which it was placed.
def place(column, player_number):
    global board, width, height
    if (column >= width or column < 0):
        return -1
    partial_height = height - 1
    while (partial_height >= 0 and board[partial_height][column] == 0):
        partial_height -= 1
    if (partial_height + 1 < height):
        board[partial_height + 1][column] = player_number
        return partial_height + 1
    return -1


# Function used to check whether game has finished or not
# Return values:
# -1 -> game tie
# 0  -> game continues
# 1  -> game won by player_number

# Possible t's:

#   010     111     10      01      100     001     101     101
#   111     010     11      11      010     010     010     010
#                   10      01      101     101     001     100

def gameFinished(player_number):
    global board, width, height
    available_moves = False
    for pos in range(width):
        if (board[height - 1][pos] == 0):
            available_moves = True
            break

    if (checkAnyT(player_number)):
        return 1

    if (not available_moves):
        return -1

    return 0


# Function for printing the game board
def printGame():
    global board, width, height
    for row in range(height - 1, -1, -1):
        for col in range(0, width):
            # if(board[x][y] == 0):
            #   print (" ")
            print (board[row][col], end=" ")
        print ("\n")
    print ("\n")


def checkAnyT(player_number):
    global board, width, height
    for r in range(0, height):
        for c in range(0, width):
            if (board[r][c] == player_number):
                if (checkWinBelow(r, c, player_number)
                    or checkWinAbove(r, c, player_number)
                    or checkLeft(r, c, player_number)
                    or checkRight(r, c, player_number)
                    or checkWinBottomRight(r, c, player_number)
                    or checkWinBottomLeft(r, c, player_number)
                    or checkWinTopLeft(r, c, player_number)
                    or checkWinTopRight(r, c, player_number)):
                    return True
    return False


def checkWinBelow(row, col, player_number):
    global board, width, height
    if (col + 1 == width or col == 0 or row + 1 == height): return False
    if (board[row + 1][col - 1] == board[row + 1][col] == board[row + 1][
            col + 1] == player_number): return True
    return False


def checkWinAbove(row, col, player_number):
    global board, width, height
    if (col == 0 or row == 0 or col + 1 == width): return False
    if (board[row - 1][col - 1] == board[row - 1][col] == board[row - 1][
            col + 1] == player_number): return True
    return False


def checkLeft(row, col, player_number):
    global board, width, height
    if (row + 1 >= height or col + 1 >= width or col - 1 < 0): return False
    if (board[row - 1][col + 1] == board[row][col + 1] == board[row + 1][col + 1] == player_number): return True
    return False


def checkRight(row, col, player_number):
    global board, width, height
    if (col == 0 or row == 0 or row + 1 == height): return False
    if (board[row - 1][col - 1] == board[row][col - 1] == board[row + 1][col - 1] == player_number): return True
    return False


def checkWinBottomRight(row, col, player_number):
    global board, width, height
    if (row + 2 >= height or col - 2 < 0): return False
    if (board[row][col -2] == player_number and board[row + 1][col - 1] == player_number and board[row + 2][
            col] == player_number): return True
    return False


def checkWinBottomLeft(row, col, player_number):
    global board, width, height
    if (row + 2 >= height or col + 2 >= width): return False
    if (board[row + 2][col] == player_number and board[row + 1][col + 1] == player_number and board[row][
            col + 2] == player_number): return True
    return False


def checkWinTopLeft(row, col, player_number):
    global board, width, height
    if (row - 2 < 0 or col + 2 >= width): return False
    if (board[row - 2][col] == player_number and board[row - 1][col + 1] == player_number and board[row][
            col + 2] == player_number): return True
    return False


def checkWinTopRight(row, col, player_number):
    global board, width, height
    if (row - 2 < 0 or col - 2 < 0): return False
    if (board[row - 2][col] == player_number and board[row - 1][col - 1] == player_number and board[row][
            col - 2] == player_number): return True
    return False


def intelligentFunction2(turn, board):
    move = int(input('Enter your move!'))
    return move 


def intelligentFunction1(turn, board):
    move = ConectaBot(board,2)
    return move 

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
    print("")
    return r


def main():
    global board
    turn = 1
    loser = 0
    while (gameFinished(turn) == 0):
        printGame()
        if (turn == 1):
            turn = 2
        else:
            turn = 1
        if (turn == 1):
            column = intelligentFunction1(turn, board)
        if (turn == 2):
            column = intelligentFunction2(turn, board)
        if (place(column, turn) == -1):
            loser = turn
            break

    # Game is a tie
    if (gameFinished(turn) == -1): print("The game is a tie!")
    elif not (loser == 0): print ("The loser is ", turn)
    else:
        printGame()
        print ("The winner is ", turn)


if __name__ == '__main__':
    main()