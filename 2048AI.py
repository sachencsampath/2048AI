import random
import numpy as np
from multiprocessing import Pool

# function to initialize game / grid
def startGame():
    # declaring an empty list then
    # appending 4 list each with four
    # elements as 0.
    matrix = []
    for i in range(4):
        matrix.append([0] * 4)
    # Transfomring Matrix to numpy array        
    matrix = np.array(matrix)
 
    # calling the function to add
    # a new 2 in grid after every step
    addNewTile(matrix)
    return matrix

# function to add a new 2 in
# grid at any random empty cell
def addNewTile(matrix):
    #find empty cells
    row, col = np.where(matrix == 0)
 
    # choosing a random index 
    randIndex = random.randint(0,row.size - 1)
    
    # we will place a 2 or a 4 at that empty
    # random cell.
    randomNum = random.randint(0, 9)
    if(randomNum == 0):
        matrix[row[randIndex]][col[randIndex]] = 4
    else:
        matrix[row[randIndex]][col[randIndex]] = 2    

# function to get the current
# state of game
def get_current_state(matrix):
 
    # if any cell contains
    # 2048 we have won
    if 4096 in matrix:
        return 'WON'
 
    # if we are still left with
    # atleast one empty cell
    # game is not yet over
    if 0 in matrix:
        return 'GAME NOT OVER'
 
    # or if no cell is empty now
    # but if after any move left, right,
    # up or down, if any two cells
    # gets merged and create an empty
    # cell then also game is not yet over
    for i in range(3):
        for j in range(3):
            if(matrix[i][j]== matrix[i + 1][j] or matrix[i][j]== matrix[i][j + 1]):
                return 'GAME NOT OVER'
 
    for j in range(3):
        if(matrix[3][j]== matrix[3][j + 1]):
            return 'GAME NOT OVER'
 
    for i in range(3):
        if(matrix[i][3]== matrix[i + 1][3]):
            return 'GAME NOT OVER'
 
    # else we have lost the game
    return 'LOST'

def compress(grid):
    changed = False
    compressed = []
    for i in range(4):
        compressed.append([0] * 4)
    compressed = np.array(compressed)  
    for i in range(4):
        pos = 0
        for j in range(4):
            if(grid[i][j] != 0):
                compressed[i][pos] = grid[i][j]
                 
                if(j != pos):
                    changed = True
                pos += 1 
    return compressed, changed

def merge(grid): 
    changed = False     
    for i in range(4):
        for j in range(3):
            if(grid[i][j] == grid[i][j + 1] and grid[i][j] != 0):
                grid[i][j] = grid[i][j] * 2
                grid[i][j + 1] = 0
                changed = True 
    return grid, changed

def reverse(grid):
    return np.flip(grid)
 
def transpose(grid):
    return np.transpose(grid)

def moveUp(grid):
    up = transpose(grid)
    up, changed = moveLeft(up)
    up = transpose(up)
    return up, changed
 
def moveDown(grid):
    down = transpose(grid)
    down, changed = moveRight(down)
    down = transpose(down)
    return down, changed

def moveLeft(grid):
    left, changed1 = compress(grid)
    left, changed2 = merge(left)
    changed = changed1 or changed2
    left, ignore = compress(left)
    return left, changed
 
def moveRight(grid):
    right = reverse(grid)
    right, changed = moveLeft(right)
    right = reverse(right)
    return right, changed

def move(grid,move):
    matrix = np.copy(grid)
    if(move == 0):
        matrix, flag = moveUp(grid)
    elif (move == 1):
        matrix, flag = moveDown(grid)
    elif (move == 2):
        matrix, flag = moveLeft(grid)   
    elif (move == 3):
        matrix, flag = moveRight(grid)
    return matrix

snakePower0 = np.array([[4**16,4**15,4**14,4**13],[4**9,4**10,4**11,4**12],[4**8,4**7,4**6,4**5],[4**1,4**2,4**3,4**4]])
snakePower1 = np.array([[4**13,4**14,4**15,4**16],[4**12,4**11,4**10,4**9],[4**5,4**6,4**7,4**8],[4**4,4**3,4**2,4**1]])
snakePower2 = np.array([[4**1,4**2,4**3,4**4],[4**8,4**7,4**6,4**5],[4**9,4**10,4**11,4**12],[4**16,4**15,4**14,4**13]])
snakePower3 = np.array([[4**4,4**3,4**2,4**1],[4**5,4**6,4**7,4**8],[4**12,4**11,4**10,4**9],[4**13,4**14,4**15,4**16]])
snakePower4 = np.array([[4**1,4**8,4**9,4**16],[4**2,4**7,4**10,4**15],[4**3,4**6,4**11,4**14],[4**4,4**5,4**12,4**13]])
snakePower5 = np.array([[4**16,4**9,4**8,4**1],[4**15,4**10,4**7,4**2],[4**14,4**11,4**6,4**3],[4**13,4**12,4**5,4**4]])
snakePower6 = np.array([[4**13,4**12,4**5,4**4],[4**14,4**11,4**6,4**3],[4**15,4**10,4**7,4**2],[4**16,4**9,4**8,4**1]])
snakePower7 = np.array([[4**4,4**5,4**12,4**13],[4**3,4**6,4**11,4**14],[4**2,4**7,4**10,4**15],[4**1,4**8,4**9,4**16]])
def heuristic(grid) -> int: 
    return max(np.sum(grid*snakePower0), np.sum(grid*snakePower1), np.sum(grid*snakePower2), np.sum(grid*snakePower3), np.sum(grid*snakePower4), np.sum(grid*snakePower5), np.sum(grid*snakePower6), np.sum(grid*snakePower7))

def minMax(grid, depth, state):
    if(depth == 0):
        return heuristic(grid)
    
    #   Max
    if(state == False):
        maxValues = [move(grid, 0),move(grid, 1),move(grid, 2),move(grid, 3)]
        score = -np.inf
        for values in maxValues:
            score = max(score, minMax(values, depth-1, True))           
        return score
    
    # min    
    if(state == True):
        #find empty cells
        row, col = np.where(grid == 0)
        score = np.inf
        minValues = []
        for index in range(row.size):
            matrix2 = np.copy(grid)
            matrix2[row[index]][col[index]] = 2
            minValues.append(matrix2)
            matrix4 = np.copy(grid)
            matrix4[row[index]][col[index]] = 4
            minValues.append(matrix4)
            
        for values in minValues:
            score = min(score,minMax(values, depth-1, False))
        return score
    
def expectiMax(grid, depth, state):
    if(depth == 0):
        return heuristic(grid)
    
    # Max
    if(state == False):
        maxValues = [move(grid, 0),move(grid, 1),move(grid, 2),move(grid, 3)]
        heuristicScore = -np.inf
        for values in maxValues:
            heuristicScore = max(heuristicScore, expectiMax(values, depth-1, True))           
        return heuristicScore
    
    # Random   
    if(state == True):
        #find empty cells
        row, col = np.where(grid == 0)
        numEmpty = row.size
        random2 = .9*(1/numEmpty)
        random4 = .1*(1/numEmpty)
        randomValues = []
        heuristicScore = 0
        for index in range(numEmpty):
            matrix2 = np.copy(grid)
            matrix2[row[index]][col[index]] = 2
            randomValues.append((matrix2,2))
            matrix4 = np.copy(grid)
            matrix4[row[index]][col[index]] = 4
            randomValues.append((matrix4,4))
            
        for values in randomValues:
            if(values[1] == 2):
                heuristicScore += random2*expectiMax(values[0], depth-1, False)
            elif(values[1] == 4):
                heuristicScore += random4*expectiMax(values[0], depth-1, False)
        return heuristicScore
        

def NextMove(Grid: list[list[int]], Step: int)->int:
    bestScore = 0
    bestMove = 0
    depth = 3
    
    for moves in range (4):
        matrix = move(Grid, moves)
            
        if(np.count_nonzero(matrix == 0) == 0):
            continue
        
        score = expectiMax(matrix, depth - 1, True)
        
        if(score > bestScore):
            bestMove = moves
            bestScore = score
        
    return bestMove 

# # calling start_game function
# # to initialize the matrix
score = 0
simulations = 0
for sims in range(10):
    matrix = startGame()
    while(True):
        x = str(NextMove(matrix,1))

        # we have to move up
        if(x == '0' or x == 'w'):

            # call the move_up function
            matrix, flag = moveUp(matrix)

            # get the current state
            status = get_current_state(matrix)

            # if game not ove then continue
            # and add a new two
            if(status == 'GAME NOT OVER' and 0 in matrix):
                addNewTile(matrix)

            # else break the loop
            else:
                break

        # the above process will be followed
        # in case of each type of move
        # below

        # to move down
        elif(x == '1' or x == 's'):
            matrix, flag = moveDown(matrix)
            status = get_current_state(matrix)
            if(status == 'GAME NOT OVER' and 0 in matrix):
                addNewTile(matrix)
            else:
                break

        # to move left
        elif(x == '2' or x == 'a'):
            matrix, flag = moveLeft(matrix)
            status = get_current_state(matrix)
            if(status == 'GAME NOT OVER' and 0 in matrix):
                addNewTile(matrix)
            else:
                break

        # to move right
        elif(x == '3' or x == 'd'):
            matrix, flag = moveRight(matrix)
            status = get_current_state(matrix)
            if(status == 'GAME NOT OVER' and 0 in matrix):
                addNewTile(matrix)
            else:

                break
        else:
            print("Invalid Key Pressed")

    print("Finished Simulation: " + str(simulations))
    print("Score: " + str(np.max(matrix)))
    simulations += 1
    score += np.max(matrix)
print("Simulations: " + str(simulations))
print("Avg Score: " + str(score / simulations)) 
    
