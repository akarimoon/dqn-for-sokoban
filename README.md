# Q-Learning for Sokoban Game

## What is "Sokoban"?
Sokoban is a [game](https://en.wikipedia.org/wiki/Sokoban) where the player pushes the crates around the warehouse, trying to get them
to goal, their "storage locations".

## What did I do?
I implemented basic Q learning algorithm using keras and visualizing using [pyxel](https://github.com/kitao/pyxel).

Note that the model and hyperparameters in the codes are not optimal (as of 11/07/19).

## Algorithm
I will leave the description of what a Q-learning to other websites and books. So as we all know, the update equation for Q-learning is,

Q(s_t,a_t)=Q(s_t,a_t)+alpha*(r_{t+1}+lambda* max{Q(s_{t+1},a)}-Q(s_t,a_t))

which can be written in code as,
```
targets[i, action] += self.alpha * (reward + self.discount * Q_sa - targets[i, action])
```
where targets contain the Q values, action is the action we take, self.alpha is the learning rate, self.discount is the discount rate, and Q_{sa} is our predicted reward from our next action.

## Model
Model is defined in `model.py` by `IB9Net` function as,
```
def IB9Net(maze_shape, lr=0.001):
    maze_size = np.product(maze_shape)
    model = Sequential()
    model.add(Reshape((maze_shape[0], maze_shape[1], 1), input_shape=(maze_size, )))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(maze_shape[0], maze_shape[1], 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=maze_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')

    return model
```
The input shape is the map (maze) shape and the output shape is 4, the number of actions that we can choose from, (left, up, right, down).

## Player and Boxes
I have named the player, __IB9__.

<img src="/images/ib9.png" width="320px">

He will be trying to learn throughout this code. Although the implemented map has only one box, it is possible to set multiple boxes. Our objective is to move all the boxes (the yellow box) to the goal (the blue box). See below for image.

<img src="/images/box.png" width="160px">
<img src="/images/goal.png" width="160px">

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
 Then, it will run a number of epochs of training as written in `play_game` function. This will store the wins and losses as well as train our model. The map we will use looks something like this (see below).

<img src="/images/map.png" width="320px">

## View found route
After training is finished, it will choose one route that resulted in a win (or choose the last game resulting in a loss). By pressing "R", you will be able to see how our beloved IB9 tries to work its way through the map.

## Hyperparameters
The hyperparameters are adjustable in lines 553-570. `hyperparams` handles hyperparameters and `reward_dict` defines the reward for each action it takes.

### Hyperparameters
- `num_epochs`: Number of epochs
- `discount`: Discount rate for updating Q value
- `epsilon`: Rate IB9 will take a random action (exploration rate)
- `min_reward`: If the total reward goes below this value, it will automatically move on to next epoch
- `min_reward_decay`: Optiion of whether implementing decay rate for `min_reward`
- `max_memory`: Number of episodes IB9 will remember while training
- `lr_rate`: Learning rate for updating Q values
- `model_lr_rate`: Learning rate for `IB9Net` (currently not used)
- `model_num_epochs`: Number of epochs when fitting `IB9Net`
- `model_batch_size`: Batch size for fitting `IB9Net`

### Rewards
- `visited`: Reward when visiting a cell that has already visited before, should be <0
- `invalid`: Reward when trying to make an invalid move (move towards the wall), should be <0
- `val_move`: Reward when making any move, should be <0 to prevent wandering but kept small
- `val_move_box`: Reward when moving the box, should be >0
- `goal`: Reward when IB9 was able to solve

Example:
```
hyperparams = {
    'num_epochs': 100,
    'discount': 0.9,
    'epsilon': 0.1,
    'min_reward': -10,
    'min_reward_decay': None,
    'max_memory': 30,
    'lr_rate': 0.01,
    'model_lr_rate': 0.001,
    'model_num_epochs': 20,
    'model_batch_size': 16
    }

reward_dict = {
    'visited': -0.7,
    'invalid': -0.8,
    'val_move': -0.04,
    'val_move_box': +0.6,
    'goal': +1.0
}
```

## Parser arguments
- no additional arguments (default): run training
- "-t" or "--train": same as default
- "-f" or "--free": free play mode (__NOT COMPLETE__)
- "-c" or "-call_pretrained": Use pretrained h5 file, specify h5 file

## Requirements
- numpy
- keras
- pyxel (python 3.7 or above required)

## Confirmed issues
If any PyxelErrors about SDL audio occurs, comment out and forth `pyxel.sound(0)` in Maze `__init__`.
Also, pyxel window will be black during training.

## References
- Deep Reinforcement Learning for Maze Solving. https://samyzaf.com/ML/rl/qmaze.html
