from random import sample, choice, random
from numpy import array, float32, rot90

WALL = 1.0
MY_HEAD = -1.0
# mutipliers
HUNGER_m = 0.01
SNAKE_m = 0.02
HEAD_m = 0.04

class Game:
    
    def __init__(self, ID, height = 11, width = 11, snake_cnt = 4, health_dec = 1):
        self.id = ID
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.health_dec = health_dec
        
        # standard starting board positions (in order) for 7x7, 11x11, and 19x19
        # battlesnake uses random positions for any non-standard board size
        # https://github.com/BattlesnakeOfficial/engine/blob/master/rules/create.go
        positions = sample(
            (
                (1, 1), (height - 2, width - 2), (height - 2, 1), (1, width - 2),
                (1, width//2), (height//2, width - 2), (height - 2, width//2), (height//2, 1)
            ),
            snake_cnt)
        self.last_moves = {i: choice((0, 1, 2, 3)) for i in range(snake_cnt)}
        
        # I changed the data structure to speed up the game
        # empty_positions is used to generate food randomly
        self.empty_positions = {(y, x) for y in range(height) for x in range(width)}
        
        self.snakes = [Snake(ID, 100, [positions[ID]] * 3) for ID in range(snake_cnt)]
        for snake in self.snakes:
            self.empty_positions.remove(snake.head.position)
        
        self.food = set(sample(self.empty_positions, snake_cnt))
        for food in self.food:
            self.empty_positions.remove(food)
        
        # two board sets are used to reduce run time
        self.heads = {snake.head.position: {snake} for snake in self.snakes}
        self.bodies = {body for snake in self.snakes for body in snake}
        
        # log
        self.wall_collision = 0
        self.body_collision = 0
        self.head_collision = 0
        self.starvation = 0
        self.food_eaten = 0
        self.game_length = 0
    
    def get_states_and_ids(self):
        states = [self.make_state(snake, self.last_moves[snake.id]) for snake in self.snakes]
        ids = [(self.id, snake.id) for snake in self.snakes]
        return (states, ids)
    
    def tic(self, moves, show = False):
        snakes = self.snakes
        # execute moves
        for i in range(len(snakes)):
            snake = snakes[i]
            move = (moves[i] + self.last_moves[snake.id] - 1) % 4
            self.last_moves[snake.id] = move
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
            snake.health -= self.health_dec
        
        # check for food eaten
        for snake in snakes:
            if snake.head.position in self.food:
                food = snake.head.position
                self.food.remove(food)
                snake.health = 100
                snake.grow()
                self.food_eaten += 1
        
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
        
        if show:
            self.draw()
        
        # remove dead snakes
        kills = set()
        for snake in snakes:
            head = snake.head.position
            # check for wall collisions
            if head[0] < 0 or head[0] >= self.height or head[1] < 0 or head[1] >= self.width:
                kills.add(snake)
                self.wall_collision += 1
            # check for body collisions
            elif head in self.bodies:
                kills.add(snake)
                self.body_collision += 1
            # check for head on collisions
            elif len(self.heads[head]) > 1:
                for s in self.heads[head]:
                    if snake.length <= s.length and s != snake:
                        kills.add(snake)
                        self.head_collision += 1
                        break
            # check for starvation
            elif snake.health <= 0:
                kills.add(snake)
                self.starvation += 1
        # remove from snakes set
        for snake in kills:
            # update board sets
            head = snake.head.position
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
            for body in snake:
                # it is possible that a snake has eaten on its first move and then die on its second move
                # in that case the snake will have a repeated tail
                # removing it from bodies twice causes an error
                # tried to debug this one for 5 hours and finally got it
                try:
                    self.bodies.remove(body)
                    self.empty_positions.add(body)
                except KeyError:
                    pass
            snakes.remove(snake)
        
        if show:
            self.draw()
        
        self.game_length += 1
        # return the winner if there is one
        if len(snakes) <= 1:
            return snakes[0].id if snakes else None
        # return -1 if the game continues
        else:
            return -1
    
    def make_state(self, you, last_move):
        """ Process the data and translate them into a grid
        
        Args:
            you: a Snake object define by snake.py; represents this snake
            last_move: the last move you made; one of {0, 1, 2, 3}
        
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
        length_minus_half = you.length - 0.5
        for snake in self.snakes:
            # get the head
            board[snake.head.position[0]][snake.head.position[1]][0] = (snake.length - length_minus_half) * HEAD_m
            # get the body
            # the head is also counted as a body for the making of the state because it will be a body next turn
            # going backwards because there could be a repeated tail when snake eats food
            body = snake.tail
            dist = 1
            while body:
                board[body.position[0]][body.position[1]][1] = dist * SNAKE_m
                body = body.prev_node
                dist += 1
        
        for food in self.food:
            board[food[0]][food[1]][2] = (101 - you.health) * HUNGER_m
        
        # from this point, all positions are measured relative to our head
        head_y, head_x = you.head.position
        board[head_y][head_x][0] = MY_HEAD
        for y in range(self.height):
            for x in range(self.width):
                grid[y - head_y + center_y][x - head_x + center_x] = board[y][x]
        
        # k = 0 => identity
        # k = 1 => rotate left
        # k = 2 => rotate 180
        # k = 3 => rotate right
        return rot90(array(grid, dtype = float32), k = last_move)
    
    def draw(self):
        board = [[0] * self.width for _ in range(self.height)]
        
        for food in self.food:
            board[food[0]][food[1]] = 9
        
        for snake in sorted(self.snakes, key = lambda s: s.length):
            # head might go out of bound
            head_y, head_x = snake.head.position
            if head_y >= 0 and head_y < self.height and head_x >= 0 and head_x < self.width:
                board[snake.head.position[0]][snake.head.position[1]] = -(snake.id + 1)
        for snake in self.snakes:
            for body in snake:
                board[body[0]][body[1]] = snake.id + 1
        
        f = open("replay.rep", 'a')
        for row in board:
            f.write(str(row) + '\n')
        f.write('\n')
        f.close()

class Snake:
    
    def __init__(self, ID, health, head_and_body):
        self.id = ID
        self.health = health
        self.length = len(head_and_body)
        self.head = Node(head_and_body[0])
        self.tail = self.head
        for i in range(1, len(head_and_body)):
            new_node = Node(head_and_body[i])
            new_node.prev_node = self.tail
            self.tail.next_node = new_node
            self.tail = new_node
    
    # iterate through the body's position (not including the head)
    def __iter__(self):
        self.curr = self.head.next_node
        return self
    
    def __next__(self):
        if self.curr:
            position = self.curr.position
            self.curr = self.curr.next_node
            return position
        else:
            raise StopIteration
    
    def move(self, direction):
        if direction == 0:   # up
            y = self.head.position[0] - 1
            x = self.head.position[1]
        elif direction == 1: # right
            y = self.head.position[0]
            x = self.head.position[1] + 1
        elif direction == 2: # down
            y = self.head.position[0] + 1
            x = self.head.position[1]
        # if direction == 3: # left
        else:
            y = self.head.position[0]
            x = self.head.position[1] - 1
        new_head = Node((y, x))
        new_head.next_node = self.head
        self.head.prev_node = new_head
        old_head = self.head
        self.head = new_head
        old_tail = self.tail
        self.tail = self.tail.prev_node
        self.tail.next_node = None
        
        # return the new head, the old head and the removed tail
        # tells the Game how to up date the board sets
        # don't remove the tail if it is on top of another body
        if old_tail.position == self.tail.position:
            old_tail.position = None
        
        return (new_head.position, old_head.position, old_tail.position)
    
    def grow(self):
        self.length += 1
        new_tail = Node(self.tail.position)
        new_tail.prev_node = self.tail
        self.tail.next_node = new_tail
        self.tail = new_tail

class Node:
    
    def __init__(self, yx):
        self.position = yx
        self.prev_node = None
        self.next_node = None
