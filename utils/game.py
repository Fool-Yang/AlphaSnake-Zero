from random import sample, choice, random
from numpy import array, rot90

from snake import Snake

WALL = 1.0
# mutipliers
HUNGER_m = 0.01
SNAKE_m = 0.02
HEAD_m = 0.04
MY_HEAD = -1.0


class Game:
    
    def __init__(self, height, width, snake_cnt):
        
        # standard starting board positions (in order) for 7x7, 11x11, and 19x19
        # battlesnake uses random positions for any non-standard board size
        # https://github.com/BattlesnakeOfficial/engine/blob/master/rules/create.go
        positions = sample(
            [
                (1, 1), (height - 2, width - 2), (height - 2, 1), (1, width - 2),
                (1, width//2), (height//2, width - 2), (height - 2, width//2), (height//2, 1)
            ],
            snake_cnt)
        
        # I changed the data structure to speed up the game
        # empty_positions is used to generate food randomly
        self.empty_positions = {(y, x) for y in range(height) for x in range(width)}
        
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        
        self.snakes = [Snake(ID, 100, [positions[ID]] * 3) for ID in range(snake_cnt)]
        for snake in self.snakes:
            self.empty_positions.remove(snake.body[0])
        
        self.food = set(sample(self.empty_positions, snake_cnt))
        for food in self.food:
            self.empty_positions.remove(food)
        
        # two board sets are used to reduce run time
        self.heads = {snake.body[0]: {snake} for snake in self.snakes}
        self.bodies = {snake.body[i] for snake in self.snakes for i in range(1, len(snake.body))}
    # game rules
    # https://github.com/BattlesnakeOfficial/rules/blob/master/standard.go
    # this link below is what they use for the engine
    # they have defferent algorithms, resulting in different rules
    # https://github.com/BattlesnakeOfficial/engine/blob/master/rules/tick.go
    # I am using the online version (first one)
    def run(self, Alice, Bob=None, sep=None):
        if Bob:
            snake_ids1 = list(range(sep))
            snake_ids2 = list(range(sep, self.snake_cnt))
            last_moves1 = {i: choice((0, 1, 2, 3)) for i in range(sep)}
            last_moves2 = {i: choice((0, 1, 2, 3)) for i in range(sep, self.snake_cnt)}
        else:
            snake_ids = list(range(self.snake_cnt))
            last_moves = {i: choice((0, 1, 2, 3)) for i in range(self.snake_cnt)}
        
        snakes = self.snakes
        # game procedures
        while len(snakes) > 1:
            
            # ask for moves
            if Bob:
                # to speed up the competing process
                # the team with more snakes left wins
                if len(snake_ids1) == 0:
                    return sep
                if len(snake_ids2) == 0:
                    return 0
                states1 = array([self.make_state(snake, last_moves1[snake.id]) for snake in snakes if snake.id < sep])
                states2 = array([self.make_state(snake, last_moves2[snake.id]) for snake in snakes if snake.id >= sep])
                moves1 = Alice.make_moves(states1, snake_ids1)
                moves2 = Bob.make_moves(states2, snake_ids2)
                i = 0
                j = 0
            else:
                states = array([self.make_state(snake, last_moves[snake.id]) for snake in snakes])
                # moves are relative to last move: turn left, go straight, or turn right
                moves = Alice.make_moves(states, snake_ids)
                i = 0
            
            # execute moves
            for snake in snakes:
                if Bob:
                    if snake.id < sep:
                        move = (moves1[i] + last_moves1[snake.id] - 1) % 4
                        last_moves1[snake.id] = move
                        i += 1
                    else:
                        move = (moves2[j] + last_moves2[snake.id] - 1) % 4
                        last_moves2[snake.id] = move
                        j += 1
                else:
                    move = (moves[i] + last_moves[snake.id] - 1) % 4
                    last_moves[snake.id] = move
                    i += 1
                # make the move
                new_head, old_head, tail = snake.move(move)
                # update board sets
                try:
                    # several heads might come to the same cell
                    self.heads[new_head].add(snake)
                except KeyError:
                    self.heads[new_head] = {snake}
                    # if it goes into an empty cell
                    if new_head in self.empty_positions:
                        self.empty_positions.remove(new_head)
                if len(self.heads[old_head]) == 1:
                    del self.heads[old_head]
                else:
                    self.heads[old_head].remove(snake)
                self.bodies.add(old_head)
                if tail:
                    self.bodies.remove(tail)
                    # no one enters this cell
                    if tail not in self.heads:
                        self.empty_positions.add(tail)
            
            # reduce health
            for snake in snakes:
                snake.health -= 1
            
            # remove dead snakes
            # I have checked the code of the battlesnake game
            # their algorithm for checking collisions is shit
            # they run a nested for loop for every snake
            # this whole check through runs in O(n) time
            kills = set()
            for snake in snakes:
                head = snake.body[0]
                # check for wall collisions
                if head[0] < 0 or head[0] >= self.height or head[1] < 0 or head[1] >= self.width:
                    kills.add(snake)
                # check for body collisions
                elif head in self.bodies:
                    kills.add(snake)
                # check for head on collisions
                elif len(self.heads[head]) > 1:
                    for s in self.heads[head]:
                        if len(snake.body) <= len(s.body) and s != snake:
                            kills.add(snake)
                            break
                # check for starvation
                elif snake.health <= 0:
                    kills.add(snake)
            # remove from snakes set
            for snake in kills:
                # update board sets
                head = snake.body[0]
                if len(self.heads[head]) == 1:
                    del self.heads[head]
                    # it might die due to starvation or equal-length head on collision
                    # only in those two cases, the head position should become an empty space
                    # not out of bound and not into a body and not into a food
                    if head[0] >= 0 and head[0] < self.height and head[1] >= 0 and head[1] < self.width:
                        # head is in range
                        if head not in self.bodies and head not in self.food:
                            self.empty_positions.add(head)
                else:
                    self.heads[head].remove(snake)
                for i in range(1, len(snake.body)):
                    b = snake.body[i]
                    # it is possible that a snake has eaten on its first move and then die on its second move
                    # in that case the snake will have a repeated tail
                    # removing it from bodies twice causes an error
                    # tried to debug this one for 5 hours and finally got it
                    try:
                        self.bodies.remove(b)
                        self.empty_positions.add(b)
                    except KeyError:
                        pass
                snakes.remove(snake)
                if Bob:
                    if snake.id < sep:
                        snake_ids1.remove(snake.id)
                    else:
                        snake_ids2.remove(snake.id)
                else:
                    snake_ids.remove(snake.id)
            
            # check for food eaten
            for snake in snakes:
                if snake.body[0] in self.food:
                    food = snake.body[0]
                    self.food.remove(food)
                    snake.health = 100
                    snake.grow()
            
            # spawn food
            if len(self.food) == 0:
                chance = 1.0
            else:
                chance = 0.15
            if random() <= chance:
                try:
                    food = choice(tuple(self.empty_positions))
                    self.food.add(food)
                    self.empty_positions.remove(food)
                except IndexError:
                    # Cannot choose from an empty set
                    pass
        
        # return the winner if there is one
        return tuple(snakes)[0].id if snakes else None
    
    def make_state(self, you, last_move):
        """ Process the data and translate them into a grid
        
        Args:
            you: a Snake object define by snake.py; represents this snake
        
        Return:
            grid: a grid that represents the game
        
        """
        
        # gotta do the math to recenter the grid
        width = self.width * 2 - 1
        height = self.height * 2 - 1
        grid = [[[0.0, WALL, 0.0] for col in range(width)] for row in range(height)]
        center_y = height//2
        center_x = width//2
        # the original game board
        # it's easier to work on the original board then transfer it onto the grid
        board = [[[0.0, 0.0, 0.0] for col in range(self.width)] for row in range(self.height)]
        
        # positions are (y, x) not (x, y)
        # because you read the grid row by row, i.e. (row number, column number)
        # otherwise the board is transposed
        length_minus_half = len(you.body) - 0.5
        for snake in self.snakes:
            body = snake.body
            # get head
            board[body[0][0]][body[0][1]][0] = (len(body) - (length_minus_half)) * HEAD_m
            # get the rest of the body
            dist = 1
            # Don't do the body[-1:0:-1] slicing. It will copy the list
            for i in range(len(body) - 1, 0, -1):
                board[body[i][0]][body[i][1]][1] = dist * SNAKE_m
                dist += 1
        
        for food in self.food:
            board[food[0]][food[1]][2] = (101 - you.health) * HUNGER_m
        
        # get my head
        head_y, head_x = you.body[0]
        board[head_y][head_x] = [MY_HEAD] * 3
        
        # from this point, all positions are measured relative to our head
        for y in range(self.height):
            for x in range(self.width):
                grid[y - head_y + center_y][x - head_x + center_x] = board[y][x]
        
        # k = 0 => identity
        # k = 1 => rotate left
        # k = 2 => rotate 180
        # k = 3 => rotate right
        return rot90(array(grid), k = last_move)