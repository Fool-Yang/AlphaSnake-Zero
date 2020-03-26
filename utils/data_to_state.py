from numpy import array, rot90

WALL = 1.0
# mutipliers
HUNGER_m = 0.01
SNAKE_m = 0.02
HEAD_m = 0.04
MY_HEAD = -1.0

def make_state(data, last_move):
    width = data['board']['width']
    height = data['board']['height']
    you = data['you']
    # the original game board
    # it's easier to work on the original board then transfer it onto the grid
    board = [[[0.0, 0.0, 0.0] for col in range(width)] for row in range(height)]
    
    # positions are (y, x) not (x, y)
    # because you read the grid row by row, i.e. (row number, column number)
    # otherwise the board is transposed
    length_minus_half = len(you['body']) - 0.5
    for snake in data['board']['snakes']:
        body = snake['body']
        # get head
        board[body[0]['y']][body[0]['x']][0] = (len(body) - (length_minus_half)) * HEAD_m
        # get the rest of the body
        dist = 1
        # Don't do the body[-1:0:-1] slicing. It will copy the list
        for i in range(len(body) - 1, 0, -1):
            board[body[i]['y']][body[i]['x']][1] = dist * SNAKE_m
            dist += 1
    
    for food in data['board']['food']:
        board[food['y']][food['x']][2] = (101 - you['health']) * HUNGER_m
    
    # get my head
    head_y = you['body'][0]['y']
    head_x = you['body'][0]['x']
    board[head_y][head_x] = [MY_HEAD] * 3
    
    g_width = width * 2 - 1
    g_height = height * 2 - 1
    grid = [[[0.0, WALL, 0.0] for col in range(g_width)] for row in range(g_height)]
    center_y = g_height//2
    center_x = g_width//2
    # from this point, all positions are measured relative to our head
    for y in range(height):
        for x in range(width):
            grid[y - head_y + center_y][x - head_x + center_x] = board[y][x]
    
    # k = 0 => identity
    # k = 1 => rotate left
    # k = 2 => rotate 180
    # k = 3 => rotate right
    return rot90(array(grid), k = last_move)
