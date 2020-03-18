class Snake:

    def __init__(self, ID, health, body):
        self.id = ID
        self.health = health
        self.body = body

    def move(self, direction):
        body = self.body
        if direction == 0:   # up
            y = body[0][0] - 1
            x = body[0][1]
        elif direction == 1: # right
            y = body[0][0]
            x = body[0][1] + 1
        elif direction == 2: # down
            y = body[0][0] + 1
            x = body[0][1]
        # if direction == 3: # left
        else:
            y = body[0][0]
            x = body[0][1] - 1
        body.insert(0, (y, x))
        tail = body.pop()

        # return the new head, the old head and the removed tail
        # tells the Game how to up date the board sets
        # don't remove the tail if it is on top of another body
        if tail == self.body[-1]:
            tail = None

        return (body[0], body[1], tail)

    def grow(self):
        self.body.append(self.body[-1])
