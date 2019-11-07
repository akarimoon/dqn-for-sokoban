import datetime
import numpy as np
import pyxel
import math
import random
import time

from parser import parse
from model import IB9Net

"""
Define reward (values are in range [-1, 1])
- Each step: -0.04 pt
- Is wall: -0.8 pt
- Visit again: -0.25 pt

- Box at goal: +1.0 pt
- Move box: +0.02 pt
"""

class IB9:
    def __init__(self, x, y):
        #coordinates
        self.x = x
        self.y = y
        #direction
        self.v = [0, -1]

class Box:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.move = False

class Experience:
    def __init__(self, model, max_memory=100, discount=0.9):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        memory_size = len(self.memory)
        data_size = min(memory_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(memory_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

class Maze:
    def __init__(self, model, **kwargs):
        self.start_coords = kwargs.pop('start_coords', {})
        self.discount = kwargs.pop('discount', 0.1)
        self.max_memory = kwargs.pop('max_memory', 10)
        self.num_epochs = kwargs.pop('num_epochs', 100)
        self.data_size = kwargs.pop('data_size', 50)
        self.epsilon = kwargs.pop('epsilon', 0.1)
        self.model_num_epochs = kwargs.pop('model_num_epochs', 8)
        self.model_batch_size = kwargs.pop('model_batch_size', 16)
        self.reward_dict = kwargs.pop('reward_dict', None)
        self.min_reward = kwargs.pop('min_reward', -0.5 * 20)
        self.min_reward_decay = kwargs.pop('min_reward_decay', None)
        self.verbose = kwargs.pop('verbose', True)

        self.free_play = kwargs.pop('free_play', False)

        if self.reward_dict is None:
            raise ValueError('Please specify a reward dictionary')

        self.shape = (48, 48)
        self.goal = self.start_coords['goal']
        self.play = True

        self.model = model
        self.experience = Experience(self.model, max_memory=self.max_memory, discount=self.discount)

        self._reset()
        pyxel.init(64, 64, caption='soko')
        pyxel.load('soko2.pyxres')
        # pyxel.sound(0)
        if self.free_play:
            pyxel.run(self.update_simple, self.draw)
        else:
            pyxel.run(self.update, self.draw)

    def _reset(self):
        self.move_count = 0
        self.move_x = 0
        self.move_y = 0
        self.is_wall = False
        self.is_goal = False
        self.move_box = False
        self.visited = set()
        self.total_reward = 0

        #initialize player
        ib9_x, ib9_y = self.start_coords['IB9']
        self.IB9 = IB9(ib9_x, ib9_y)
        x, y = self.IB9.x, self.IB9.y
        self.state = (x, y, 'start')

        #initialize box (written as list for cases with multiple boxes)
        self.box = [Box(x, y) for x, y in self.start_coords['box']]
        self.until_count = len(self.box)

    def _check_wall(self, item, check_dirc, canvas):
        x = (item.x + check_dirc[0] * 8) // 8
        y = (item.y + check_dirc[1] * 8) // 8
        if (x, y) == (16, 24):
            return True
        try:
            return canvas[y][x] == 64
        except:
            return True

    def _check_box(self, item, box, check_dirc):
        return (item.x + check_dirc[0] * 8 == box.x) and (item.y + check_dirc[1] * 8 == box.y)

    def _check_box2(self, item, check_dirc, canvas):
        remove = False
        for box in self.box:
            if self._check_box(item, box, check_dirc):
                box.move = True
                #check if box or wall ahead of box
                for b in self.box:
                    if self._check_box(box, b, check_dirc) or self._check_wall(box, check_dirc, canvas=canvas):
                        box.move = False
                        remove = True
                        break
            else:
                box.move = False

        return remove

    # def _check_goal(self, item):
    #     return pyxel.tilemap(0).get(round(item.x / 8), round(item.y / 8)) == 3

    def _check_goal2(self, item):
        return (item.x == self.goal[0]) and (item.y == self.goal[1])

    def _map2array(self):
        tilearray = []
        for i in range(self.shape[0] // 8):
            temp = []
            for j in range(self.shape[1] // 8):
                if pyxel.tilemap(0).data[i][j] == 64:
                    temp.append(64)
                else:
                    temp.append(0)
            tilearray.append(temp)
        return np.array(tilearray)

    def get_actions(self, canvas, cell=None):
        if cell is None:
            x, y, mode = self.state
        else:
            x, y = cell
        xlim, ylim = self.shape
        actions = [0, 1, 2, 3]

        if x == 0: # if at very left, can't move left
            actions.remove(0)
        elif x == xlim - 8: # if at very right, can't move right
            actions.remove(2)

        if y == 0: # if at very top, can't move up
            actions.remove(1)
        elif y == ylim - 8: # if at very IB9tom, can't move down
            actions.remove(3)

        if self.IB9.x > 0:
            if self._check_wall(self.IB9, check_dirc=(-1, 0), canvas=canvas) or self._check_box2(self.IB9, check_dirc=(-1, 0), canvas=canvas):
                actions.remove(0)
        if  self.IB9.y > 0:
            if self._check_wall(self.IB9, check_dirc=(0, -1), canvas=canvas) or self._check_box2(self.IB9, check_dirc=(0, -1), canvas=canvas):
                actions.remove(1)
        if self.IB9.x < xlim - 8:
            if self._check_wall(self.IB9, check_dirc=(1, 0), canvas=canvas) or self._check_box2(self.IB9, check_dirc=(1, 0), canvas=canvas):
                actions.remove(2)
        if self.IB9.y < ylim - 8:
            if self._check_wall(self.IB9, check_dirc=(0, 1), canvas=canvas) or self._check_box2(self.IB9, check_dirc=(0, 1), canvas=canvas):
                actions.remove(3)

        return actions

    def get_reward(self, box_moved):
        x, y, mode = self.state
        xlim, ylim = self.shape
        if mode == 'blocked':
            return self.min_reward - 1
        if mode == 'invalid':
            return self.reward_dict['invalid']
        if (x, y) in self.visited:
            return self.reward_dict['visited']
        if mode == 'valid':
            if box_moved: return self.reward_dict['val_move_box']
            else: return self.reward_dict['val_move']

    def observe(self, get_canvas=False):
        canvas = self._map2array() # wall = 64, else = 0
        canvas[self.IB9.y // 8][self.IB9.x // 8] = 1 # IB9 = 1
        i = 1
        for box in self.box: # box = 10 * i
            canvas[box.y // 8][box.x // 8] = 10 * i
            i += 1
        if get_canvas:
            return canvas
        envstate = canvas.reshape((1, -1))
        return envstate

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        for box in self.box:
            if self._check_goal2(box):
                self.until_count -= 1
        if self.until_count == 0:
            return 'win'
        return 'not_over'

    # manage game
    def update_state(self, action):
        #add current coords to visted
        if (self.IB9.x, self.IB9.y) not in self.visited:
            self.visited.add((self.IB9.x, self.IB9.y))

        canvas = self.observe(get_canvas=True)
        actions = self.get_actions(canvas)
        if not actions:
            new_mode = 'blocked'
        elif action in actions:
            new_mode = 'valid'
            if action == LEFT:
                self.set_move(-1, 0, canvas)
            elif action == RIGHT:
                self.set_move(1, 0, canvas)
            elif action == UP:
                self.set_move(0, -1, canvas)
            elif action == DOWN:
                self.set_move(0, 1, canvas)
            dirc = (self.move_x // 8, self.move_y // 8)
            for box in self.box:
                if self._check_box(self.IB9, box, check_dirc=dirc):
                    box.move = True
                    for b in self.box:
                        if self._check_box(box, b, check_dirc=dirc) or self._check_wall(box, check_dirc=dirc, canvas=canvas):
                            box.move = False
                            self.move_count = 0
                            break
                else:
                    box.move = False
        else:
            new_mode = 'invalid'
            self.move_count = 0

        if self.move_count > 0:
            self.IB9.x += self.move_x
            self.IB9.y += self.move_y

            for box in self.box:
                if box.move:
                    box.x += self.move_x
                    box.y += self.move_y

        self.state = (self.IB9.x, self.IB9.y, new_mode)

        box_moved = 0
        for box in self.box:
            box_moved += box.move
        box_moved = box_moved > 0

        reward = self.get_reward(box_moved)
        self.total_reward += reward

        envstate = self.observe()
        status = self.game_status()

        return envstate, reward, status

    def set_move(self, x, y, canvas):
        # self.move_x = x
        # self.move_y = y
        self.move_x = x * 8
        self.move_y = y * 8
        self.IB9.v = [abs(x), x + y]
        self.move_count = 8

        if self._check_wall(self.IB9, check_dirc=(x, y), canvas=canvas):
            self.move_count = 0
            for box in self.box:
                if self._check_box(self.IB9, box, check_dirc=(x, y)):
                    box.move = True
                    for b in self.box:
                        if self._check_box(box, b, check_dirc=(x, y)) or self._check_wall(box, check_dirc=(x, y), canvas=canvas):
                            box.move = False
                            self.move_count = 0
                            break
                else:
                    box.move = False

    def play_game(self):
        start_time = datetime.datetime.now()
        win_history = []
        win_actions = []
        win_rate = 0.0
        hsize = self.shape[0] // 2
        # imctr = 1

        for epoch in range(self.num_epochs):
            loss = 0.0
            self._reset()
            game_over = False

            envstate = self.observe()

            num_episodes = 0
            action_history = []
            while not game_over:
                canvas = self.observe(get_canvas=True)
                actions = self.get_actions(canvas)
                if not actions:
                    break
                prev_envstate = envstate
                if np.random.rand() < self.epsilon:
                    action = random.choice(actions)
                else:
                    action = np.argmax(self.experience.predict(prev_envstate))

                envstate, reward, game_status = self.update_state(action)
                action_history.append(actions_dict[action])

                if game_status == 'win':
                    if self.verbose: print("Win, next epoch")
                    win_history.append(1)
                    win_actions.append(action_history)
                    if self.min_reward_decay is not None: self.min_reward *= self.min_reward_decay
                    game_over = True
                elif game_status == 'lose':
                    if self.verbose: print("Lost, next epoch")
                    win_history.append(0)
                    if epoch + 1 == self.num_epochs: _actions = action_history
                    game_over = True
                else:
                    game_over = False

                episode = [prev_envstate, action, reward, envstate, game_over]
                self.experience.remember(episode)
                num_episodes += 1

                inputs, targets = self.experience.get_data(data_size=self.data_size)

                h = model.fit(inputs, targets, epochs=self.model_num_epochs, batch_size=self.model_batch_size, verbose=0)
                loss = model.evaluate(inputs, targets, verbose=0)
                # time.sleep(3)

            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:]) / hsize

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            template = "Epoch: {}/{} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
            print(template.format(epoch + 1, self.num_epochs, loss, num_episodes, sum(win_history), win_rate, t))

            if win_rate > 0.9: self.epsilon = 0.05
            if sum(win_history[-hsize:]) == hsize:
                print("Reached 100%% win rate at epoch: %d" % (epoch,))
                break

        if len(win_actions) >= 1:
            self.one_action = random.choice(win_actions)
        else:
            self.one_action = _actions
        if self.verbose: print(self.one_action)
        self._reset()
        self.play = False

    def run_step(self, action):
        dircs = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        dirc = dircs[action]
        canvas = self.observe(get_canvas=True)
        actions = self.get_actions(canvas)
        self.set_move(dirc[0], dirc[1], canvas)
        for box in self.box:
            if self._check_box(self.IB9, box, check_dirc=dirc):
                box.move = True
                for b in self.box:
                    if self._check_box(box, b, check_dirc=dirc) or self._check_wall(box, check_dirc=dirc, canvas=canvas):
                        box.move = False
                        self.move_count = 0
                        break
            else:
                box.move = False

        # if count is > 8 then move
        if self.move_count > 0:
            self.move_count -= 1
            self.IB9.x += self.move_x
            self.IB9.y += self.move_y

            for box in self.box:
                if box.move:
                    box.x += self.move_x
                    box.y += self.move_y
        time.sleep(0.5)

    def update(self):
        while self.play:
            self.play_game()

        if pyxel.btn(pyxel.KEY_R):
            if len(self.one_action) == 0:
                print("Finished action")
                self._reset()
            print("Run action, press R to run step by step")
            action = action_str2num(self.one_action.pop(0))
            self.run_step(action)
            print(actions_dict[action])

    def update_simple(self):
        if pyxel.btn(pyxel.KEY_LEFT):
            self.move(-1, 0)
        elif pyxel.btn(pyxel.KEY_UP):
            self.move(0, -1)
        elif pyxel.btn(pyxel.KEY_RIGHT):
            self.move(1, 0)
        elif pyxel.btn(pyxel.KEY_DOWN):
            self.move(0, 1)
        elif pyxel.btn(pyxel.KEY_0):
            self._reset()

        # if count is > 8 then move
        if self.move_count > 0:
            self.move_count -= 1
            self.IB9.x += self.move_x
            self.IB9.y += self.move_y

            for box in self.box:
                if box.move:
                    box.x += self.move_x
                    box.y += self.move_y

    def move(self, x, y):
        if self.move_count == 0:
            self.move_count = 8
            self.move_x = x
            self.move_y = y
            self.IB9.v = [abs(x), x + y]

            if pyxel.tilemap(0).get(math.floor(self.IB9.x / 7) + self.move_x, math.floor(self.IB9.y / 7) + self.move_y) == 64:
                self.move_count = 0
                for box in self.box:
                    if (self.IB9.x + self.move_x * 8 == box.x) and (self.IB9.y + self.move_y * 8 == box.y):
                        box.move = True
                        for b in self.box:
                                if (box.x + self.move_x * 8 == b.x) and (box.y + self.move_y*8 == b.y)\
                                or (pyxel.tilemap(0).get(math.floor(box.x / 7) + self.move_x, math.floor(box.y / 7) + self.move_y) == 64):
                                    box.move = False
                                    self.move_count = 0
                                    break
                    else:
                        box.move = False

    def draw(self):
        #color background white
        pyxel.cls(7)
        #paste map
        pyxel.bltm(0, 0, 0, 0, 0, 8, 8, 0)
        #draw player
        pyxel.blt(self.IB9.x, self.IB9.y, 0, 8 * self.IB9.v[0], 0, 8 * self.IB9.v[1], 8 * self.IB9.v[1], 11)
        #draw goal
        pyxel.blt(self.goal[0], self.goal[1], 0, 24, 8, 8, 8, 0)
        #draw box
        for box in self.box:
            pyxel.blt(box.x, box.y, 0, 16, 8, 8, 8, 0)


        # if self.until_count == 0:
        #     pyxel.rect(11, 32, 52, 36, 15)
        #     pyxel.text(11, 32, 'Game Clear!', pyxel.frame_count % 16)
        #     self._reset()

    def draw_simple(self):
        self.clear_count = 0
        for box in self.box:
            if pyxel.tilemap(0).get(round(box.x / 8), round(box.y / 8)) == 34:
                pyxel.blt(box.x, box.y, 0, 24, 16, 8, 8, 0)
                self.clear_count += 1
            else:
                pyxel.blt(box.x, box.y, 0, 16, 16, 8, 8, 0)

        if self.clear_count == len(self.box):
            pyxel.rect(11, 32, 52, 36, 15)
            pyxel.text(11, 32, "GAME CLEAR!", pyxel.frame_count % 16)
            self._reset()

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f s" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f min" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hr" % (h,)

def action_str2num(action):
    if action == "LEFT":
        return 0
    elif action == "UP":
        return 1
    elif action == "RIGHT":
        return 2
    elif action =="DOWN":
        return 3

if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    free_play = args.free

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    # Actions dictionary
    actions_dict = {
        LEFT: 'LEFT',
        UP: 'UP',
        RIGHT: 'RIGHT',
        DOWN: 'DOWN',
    }

    hyperparams = {
        'num_epochs': 100,
        'discount': 0.9,
        'epsilon': 0.1,
        'min_reward': -10,
        'min_reward_decay': None,
        'max_memory': 30,
        'model_lr_rate': 0.001,
        'model_num_epochs': 8,
        'model_batch_size': 16
        }

    reward_dict = {
        'visited': -0.6,
        'invalid': -0.7,
        'val_move': -0.04,
        'val_move_box': +0.8
    }

    start_coords = {
        'IB9': (8, 16),
        'box': [(32, 24)],
        'goal': (40, 24)
    }

    model = IB9Net(maze_shape=(6, 6), lr=hyperparams['model_lr_rate'])

    Maze(model, num_epochs=hyperparams['num_epochs'], discount=hyperparams['discount'],
         min_reward=hyperparams['min_reward'], epsilon=hyperparams['epsilon'], reward_dict=reward_dict, free_play=free_play,
         start_coords=start_coords)
