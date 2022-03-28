import numpy as np

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
        

def NextMove(Grid, Step)->int:
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