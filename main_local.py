import json
import os
from random import choice
import bottle
from numpy import reshape

from api import ping_response, start_response, move_response, end_response
from utils.data_to_state import make_state
from utils.alphaNNet import AlphaNNet


@bottle.route('/')
def index():
    return '''
    Battlesnake documentation can be found at
       <a href="https://docs.battlesnake.com">https://docs.battlesnake.com</a>.
    '''


@bottle.route('/static/<path:path>')
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.
    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root='static/')


@bottle.post('/ping')
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()


@bottle.post('/start')
def start():
    data = bottle.request.json
    
    """
    TODO: If you intend to have a stateful snake AI,
            initialize your snake state here using the
            request's data if necessary.
    """
    # print(json.dumps(data))
    
    # See https://docs.battlesnake.com/snake-customization for customizations
    
    color = '#00FFFF'
    headType = 'bwc-scarf'
    tailType = 'freckled'
    
    return start_response(color, headType, tailType)


@bottle.post('/move')
def move():
    data = bottle.request.json
    
    """
    TODO: Using the data from the endpoint request object, your
            snake AI must choose a direction to move in.
    """
    # print(json.dumps(data))
    head_y, head_x = data['you']['body'][0]['y'], data['you']['body'][0]['x']
    next_y, next_x = data['you']['body'][1]['y'], data['you']['body'][1]['x']
    if head_y < next_y:
        last_move = 0
    elif head_y > next_y:
        last_move = 2
    elif head_x < next_x:
        last_move = 3
    elif head_x > next_x:
        last_move = 1
    else:
        last_move = choice((0, 1, 2, 3))
    state_list = [make_state(data, last_move)]
    states = reshape(state_list, (-1, len(state_list[0]), len(state_list[0][0]), 3))
    move = (last_move + argmax(AlphaSnake.v(states)[0]) - 1) % 4
    return move_response(['up', 'right', 'down', 'left'][move])


@bottle.post('/end')
def end():
    data = bottle.request.json
    
    """
    TODO: If your snake AI was stateful,
        clean up any stateful objects here.
    """
    # print(json.dumps(data))
    
    return end_response()


def argmax(z):
    if z[0] > z[1]:
        if z[0] > z[2]:
            return 0
        else:
            return 2
    else:
        if z[1] > z[2]:
            return 1
        else:
            return 2

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    global AlphaSnake
    AlphaSnake = AlphaNNet(model = "models/alphaSnake.h5")
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug=os.getenv('DEBUG', True)
    )