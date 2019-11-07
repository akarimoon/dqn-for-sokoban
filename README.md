# Q-Learning for Sokoban Game

## What is "Sokoban"?
Sokoban is a [game](https://en.wikipedia.org/wiki/Sokoban) where the player pushes the crates around the warehouse, trying to get them
to goal, their "storage locations".

## What did I do?
I implemented basic Q learning algorithm using keras and visualizing using [pyxel](https://github.com/kitao/pyxel).

Note that the model and hyperparameters in the codes are not optimal (as of 11/07/19).

## Algorithm
I will leave the description of what a Q-learning to other websites and books. So as we all know, the update equation for Q-learning is,

![equation](http://www.sciweavers.org/tex2img.php?eq=Q%28s_t%2Ca_t%29%3DQ%28s_t%2Ca_t%29%2B%5Calpha%28r_%7Bt%2B1%7D%2B%5Clambda%5Cmax_%7B%5Calpha%7DQ%28s_%7Bt%2B1%7D%2Ca%29-Q%28s_t%2Ca_t%29%29%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

which can be written in code as,
```
targets[i, action] = reward + self.discount * Q_sa
```
where targets contain the Q values, action is the action we take, self.discount is the discount rate, and Q_{sa} is our predicted reward from our next action.

## Model
Model is defined in `model.py` by `IB9Net` function as,
```
def IB9Net(maze_shape, lr=0.001):
    maze_size = np.product(maze_shape)
    model = Sequential()
    model.add(Dense(maze_size, input_shape=(maze_size,)))
    model.add(Activation('relu'))
    model.add(Dense(maze_size))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')

    return model
```
The input shape is the map (maze) shape and the output shape is 4, the number of actions that we can choose from, (left, up, right, down).

## Player and Boxes
I have named the player, __IB9__. He will be trying to learn throughout this code. Although the implemented map has only one box, it is possible to set multiple boxes.
The starting coordinates of IB9, boxes, and goal can be set as a hyperparameters. See line 573- in `search.py`. The coordinates should be set using a dictionary and passed to class `Maze` as an argument, `start_coords`.

Example:
```
start_coords = {
    'IB9': (8, 16),
    'box': [(32, 24)],
    'goal': (40, 24)
}
```

IB9 is as for now, not able to solve the original problem,
```
start_coords = {
    'IB9': (8, 16),
    'box': [(16, 16)],
    'goal': (40, 24)
}
```
However, conditions described in example is confirmed to find a route to clear the problem.

### About the coordinates
The coordinates are all multiplied by 8 and (0, 0) is at the top left. This means that one cell to the right or down will be adding 8.

## Maze (Map)
The class `Maze` handles all visualization and training. It will first initialize the map by the `_reset` function.
 Then, it will run a number of epochs of training as written in `play_game` function. This will store the wins and losses as well as train our model.

## View found route
After training is finished, it will choose one route that resulted in a win (or choose the last game resulting in a loss). By pressing "R", you will be able to see how our beloved IB9 tries to work its way through the map.

## Hyperparameters
The hyperparameters are adjustable in lines 553-570. `hyperparams` handles hyperparameters and `reward_dict` defines the reward for each action it takes.

Example:
```
hyperparams = {
    'num_epochs': 100,
    'discount': 0.9,
    'epsilon': 0.1,
    'min_reward': -20,
    'min_reward_decay': None,
    'max_memory': 30,
    'model_lr_rate': 0.001,
    'model_num_epochs': 8,
    'model_batch_size': 16
    }

reward_dict = {
    'visited': -0.6, #if it visits a cell that has already been visited, it will be penalized by -0.6
    'invalid': -0.7, #if it tries to go towards a wall, it will be penalized by -0.7
    'val_move': -0.04, #any move costs -0.04
    'val_move_box': +0.8 #moving a box results in a reward of +0.8
}
```

## Parser arguments
- no additional arguments (default): run training
- "-t" or "--train": same as default
- "-f" or "--free": free play mode (__NOT COMPLETE__)

## Requirements
- numpy
- keras
- pyxel (python 3.7 or above required)

## Confirmed issues
If any PyxelErrors about SDL audio occurs, comment out and forth `pyxel.sound(0)` in Maze `__init__`.
Also, pyxel window will be black during training.

## References
- Deep Reinforcement Learning for Maze Solving. https://samyzaf.com/ML/rl/qmaze.html
