import numpy as np
import random
import gym
import gym.spaces as spaces
from gym.envs.classic_control import rendering
import time

class TetrisPiece:
    def __init__(self, center, template):
        self.template = np.array([list(row) for row in template], dtype=np.int8).astype(bool)
        self.cntr = center
        self.y_dim, self.x_dim = self.template.shape

I_SHAPE = [TetrisPiece(0, ["1",
                           "1",
                           "1",
                           "1"]),
           TetrisPiece(2, ["1111"])]

J_SHAPE = [TetrisPiece(1, ["100",
                           "111"]),
           TetrisPiece(0, ["11",
                           "10",
                           "10"]),
           TetrisPiece(1, ["111",
                           "001"]),
           TetrisPiece(1, ["01",
                           "01",
                           "11"])]

L_SHAPE = [TetrisPiece(1, ["001",
                           "111"]),
           TetrisPiece(0, ["10",
                           "10",
                           "11"]),
           TetrisPiece(1, ["111",
                           "100"]),
           TetrisPiece(1, ["11",
                           "01",
                           "01"])]

O_SHAPE = [TetrisPiece(1, ["11",
                           "11"])]

S_SHAPE = [TetrisPiece(1, ["011",
                           "110"]),
           TetrisPiece(0, ["10",
                           "11",
                           "01"])]

T_SHAPE = [TetrisPiece(1, ["010",
                           "111"]),
           TetrisPiece(0, ["10",
                           "11",
                           "10"]),
           TetrisPiece(1, ["111",
                           "010"]),
           TetrisPiece(1, ["01",
                           "11",
                           "01"])]

Z_SHAPE = [TetrisPiece(1, ["110",
                           "011"]),
           TetrisPiece(1, ["01",
                           "11",
                           "10"])]

#               R    G    B
WHITE       = (255, 255, 255)
GRAY        = (128, 128, 128)
BLACK       = (  0,   0,   0)
RED         = (255,   0,   0)
LIGHTRED    = (255, 128, 128)
GREEN       = (  0, 255,   0)
LIGHTGREEN  = (128, 255, 128)
BLUE        = (  0,   0, 255)
LIGHTBLUE   = (128, 128, 255)
YELLOW      = (255, 255,   0)
LIGHTYELLOW = (255, 255,  128)
COLORS      = (     BLUE,      GREEN,      RED,      YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS) # each color must have light color

class GameState:
    def __init__(self, board_size=(10,20), only_squares=False):
        # self.x_board, self.y_board = board_size
        self.Y_BOARD, self.X_BOARD = board_size
        if only_squares:
            self.TETRIMINOS = {'O': O_SHAPE}
        else:
            self.TETRIMINOS = {
                'I': I_SHAPE,
                'J': J_SHAPE,
                'L': L_SHAPE,
                'O': O_SHAPE,
                'S': S_SHAPE,
                'T': T_SHAPE,
                'Z': Z_SHAPE
            }
        self.LINE_MULTI = [0, 100, 300, 500, 800]
        self.COMBO_MULTI = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5]
        self.MAX_COMBO = len(self.COMBO_MULTI)-1

        #render stuffs
        self.BOXSIZE = 20
        self.X_SCREEN, self.Y_SCREEN = self.X_BOARD*self.BOXSIZE, self.Y_BOARD*self.BOXSIZE
        self.COLOR = 1

        self.reset()

    def reset(self):
        self.mini_board = np.zeros((self.Y_BOARD, self.X_BOARD), dtype=bool)
        self.render_board = np.zeros((self.Y_SCREEN, self.X_SCREEN, 3), dtype=np.uint8)
        self.remaining = list(self.TETRIMINOS.keys()).copy()

        self.score = 0
        self.lines = 0
        self.combo = -1
        self.level = self.calc_level()
        self.game_over = False

        self.curr_piece = self.get_new_piece() # set current piece
        self.next_piece = self.get_new_piece() # set next piece

    def get_new_piece(self):
        # refresh remaining list if it is empty
        if (len(self.remaining) <= 0):
            self.remaining = list(self.TETRIMINOS.keys()).copy()
        # get next piece
        piece = random.choice(self.remaining)
        self.remaining.remove(piece)
        return piece

    def cycle_pieces(self):
        self.curr_piece = self.next_piece
        self.next_piece = self.get_new_piece()

    def check_collision(self, piece, x, y):
        collisions = np.logical_and(piece.template, self.mini_board[y:y+piece.y_dim,x:x+piece.x_dim])
        return collisions.any()

    def place_piece(self, rotation, x, y=0, force_on=False, gravity=False):
        piece = self.TETRIMINOS[self.curr_piece]
        piece = piece[rotation%len(piece)]
        adj_x, adj_y = x-piece.cntr, y
        if force_on: # force piece left/right onto board
            adj_x = np.clip(adj_x, 0, self.X_BOARD - piece.x_dim)
        if self.check_collision(piece, adj_x, adj_y): # check if piece can be placed
            return False, -1
        if gravity: # starting from top, keep shifting piece down
            while y+adj_y+piece.y_dim < self.Y_BOARD and not self.check_collision(piece, adj_x, adj_y+1):
                adj_y += 1
        self.mini_board[adj_y:piece.y_dim+adj_y, adj_x:piece.x_dim+adj_x] |= piece.template # place piece using logical or
        # draw the individual boxes on the self.render_board - could probably parallelize this
        for x in range(piece.x_dim):
            for y in range(piece.y_dim):
                if piece.template[y][x] != 0:
                    pixelx, pixely = (x+adj_x)*self.BOXSIZE, (y+adj_y)*self.BOXSIZE
                    self.render_board[pixely+1:self.BOXSIZE+pixely, pixelx+1:self.BOXSIZE+pixelx] = COLORS[self.COLOR]
                    self.render_board[pixely+1:self.BOXSIZE-3+pixely, pixelx+1:self.BOXSIZE-3+pixelx] = LIGHTCOLORS[self.COLOR]
        return True, adj_y-y

    def remove_completed_lines(self):
        completedLines = [all(row) for row in self.mini_board]
        num_removed_lines = sum(completedLines)
        self.mini_board = np.delete(self.mini_board, completedLines, axis=0)
        self.mini_board = np.insert(self.mini_board, 0, np.zeros((num_removed_lines,self.X_BOARD), dtype=bool), axis=0)
        return num_removed_lines

    def calc_level(self): # calculate level based on the lines cleared
        self.level = min(int(self.lines / 10)+1, 15)
        return self.level

    def update_score(self, piece_fallen, cleared_lines):
        additional_score = 0
        self.combo = self.combo + 1 if cleared_lines > 0 else -1
        if self.combo > -1:
            additional_score += 50 * self.COMBO_MULTI[self.combo % self.MAX_COMBO] * self.level # updating score for combos
        additional_score += self.LINE_MULTI[cleared_lines] * self.level  # updating score for cleared lines
        additional_score += piece_fallen # how much the piece has fallen
        additional_score += cleared_lines
        self.score += additional_score
        return additional_score

    def get_board(self, render=True, put_channels_first=False):
        if render:
            # if put_channels_first:
            return np.moveaxis(self.render_board, -1, 0)
            # return self.render_board
        else:
            return self.mini_board

    def screen_dim(self):
        return (3, self.Y_SCREEN, self.X_SCREEN)

class TetrisEnv(gym.Env):
    def __init__(self, board_size=(20,10), grouped_actions=False, only_squares=False, no_rotations=False):
        super(TetrisEnv, self).__init__()
        self.state = GameState(board_size, only_squares)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.screen_dim(), dtype=np.uint8)
        self.no_rotations = no_rotations
        if no_rotations:
            self._action_set = [a for a in range(self.state.X_BOARD)]
            self.action_space = spaces.Discrete(self.state.X_BOARD)
        else:
            self._action_set = [a for a in range(self.state.X_BOARD * 4)]
            self.action_space = spaces.Discrete(self.state.X_BOARD * 4)
        # other stuff
        self.viewer = None

    def reset(self):
        self.state.reset()
        state = self.state.get_board(render=True, put_channels_first=True)
        return state

    def step(self, action=None, rand_action=False):
        if rand_action and action is None:
            action = random.choice(self._action_set)
        if action is not None:
            if self.no_rotations:
                x_index = action
                rotation = 0
            else:
                x_index = int(action/4)
                rotation = action % 4
        else:
            print("Error!")
            return
        placed_success, shifted_down = self.state.place_piece(rotation, x_index, force_on=True, gravity=True)
        if placed_success:
            cleared = self.state.remove_completed_lines()
            reward = self.state.update_score(shifted_down, cleared)
            self.state.cycle_pieces()
        else:
            reward = 0
        state = self.state.get_board(render=True, put_channels_first=True)
        return state, reward, not placed_success, {}

    def render(self, mode='image', wait_sec=0):
        image = self.state.get_board(render=True)
        if mode =='image':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(image)
            time.sleep(wait_sec)
        elif mode == 'print':
            print("Curr piece:", self.curr_piece, ", next piece:", self.next_piece)
            print("Game over:", self.game_over)
            print(self.state.get_board(render=False))
        elif mode =='rbg_array' or mode == 'human':
            return image
        else:
            print(mode)

    def close_render(self):
        if self.viewer is not None and self.viewer.isopen:
            self.viewer.close()

    def measure_step_time(self, warmup=30, steps=1000, verbose=False):
        t = time.perf_counter()
        self.reset()
        if verbose:
            print("Warming up...")
        while (time.perf_counter()-t < warmup):
            _ = env.step(rand_action=True)
        if verbose:
            print("Measuring time...")
        t = time.perf_counter()
        for _ in range(steps):
            _ = env.step(rand_action=True)
        step_per_sec = steps/(time.perf_counter()-t)
        self.reset()
        if verbose:
            print("{} steps per second".format(step_per_sec))
        return step_per_sec


if __name__ == '__main__':
    env = TetrisEnv(only_squares=False)
    env.reset()
    # env.measure_step_time(verbose=True)
    env.render(wait_sec=0.3)
    for _ in range(100):
        screen, reward, game_over, info = env.step(rand_action=True)
        env.render(wait_sec=0.3)
        if game_over:
            env.reset()
