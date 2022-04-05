import numpy as np
import random
import gym
import gym.spaces as spaces
import time

class TetrisPiece:
    def __init__(self, center, template):
        self.template = np.array([list(row) for row in template], dtype=np.int8).astype(bool)
        # self.template = np.flip(self.template)
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
DARK_GRAY   = (200, 200, 200)
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

class DrawBoard:
    def __init__(self, board_size):
        # sizes
        self.BOX_SIZE  = 20
        piece_box = self.BOX_SIZE*4
        board_shape = (board_size[0] * self.BOX_SIZE, board_size[1] * self.BOX_SIZE)
        self.EDGE_SIZE = 5
        self.Y_SCREEN  = max(board_shape[0]+self.EDGE_SIZE*2, piece_box*2+self.EDGE_SIZE*3)
        self.X_SCREEN  = board_shape[1] + self.EDGE_SIZE * 3 + piece_box
        # colors
        self.BACKGROUND_COLOR = GRAY
        self.EMPTYBOX_COLOR = BLACK
        self.PLACED_PIECE_COLOR = DARK_GRAY
        self.CURRENT_PIECE_COLOR = GREEN
        self.GHOST_PIECE_COLOR = tuple([x*0.3 for x in self.CURRENT_PIECE_COLOR])
        # slices
        self.BOARD_SLICE = np.s_[self.EDGE_SIZE:self.EDGE_SIZE+board_shape[0],self.EDGE_SIZE:self.EDGE_SIZE+board_shape[1]]
        self.CURR_SLICE = np.s_[self.EDGE_SIZE:self.EDGE_SIZE+piece_box,self.EDGE_SIZE*2+board_shape[1]:self.EDGE_SIZE*2+board_shape[1]+piece_box]
        self.NEXT_SLICE = np.s_[self.EDGE_SIZE*2+piece_box:self.EDGE_SIZE*2+piece_box*2,self.EDGE_SIZE*2+board_shape[1]:self.EDGE_SIZE*2+board_shape[1]+piece_box]
        # board
        self.EMPTY_BOARD = np.full((self.Y_SCREEN, self.X_SCREEN, 3), self.BACKGROUND_COLOR, dtype=np.uint8)
        self.EMPTY_BOARD[self.BOARD_SLICE] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.CURR_SLICE] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.NEXT_SLICE] = self.EMPTYBOX_COLOR
        # tetris box patterns
        self.PATTERN_1 = np.zeros((self.BOX_SIZE, self.BOX_SIZE), dtype=bool)
        self.PATTERN_1[1:,1:] = True
        #
        self.PREV_GHOST_SLICE = None
        self.reset()

    def reset(self):
        self.screen = self.EMPTY_BOARD.copy()

    def draw_boxes(self, squares, slice, color, transparent=False):
        square_loc = np.repeat(np.repeat(squares, 20,axis=0), 20,axis=1)
        pattern1 = np.logical_and(np.tile(self.PATTERN_1, squares.shape), square_loc)
        self.screen[slice][:pattern1.shape[0], :pattern1.shape[1]][pattern1] = color

    def draw_landed_piece(self, piece, adj_x, adj_y):
        pixelx, pixely = adj_x*self.BOX_SIZE+self.EDGE_SIZE, adj_y*self.BOX_SIZE+self.EDGE_SIZE
        slice =  np.s_[pixely:pixely+piece.y_dim*self.BOX_SIZE,pixelx:pixelx+piece.x_dim*self.BOX_SIZE]
        squares = piece.template
        self.draw_boxes(squares, slice, self.PLACED_PIECE_COLOR)
        self.PREV_GHOST_SLICE = None

    def clear_rows(self, rows):
        current_screen = self.screen[self.BOARD_SLICE]
        renderedCompletedLines = np.repeat(rows, self.BOX_SIZE, axis=0)
        num_rendered_completed_lines = sum(renderedCompletedLines)
        current_screen = np.delete(current_screen, renderedCompletedLines, axis=0)
        current_screen = np.insert(current_screen, 0, np.zeros((num_rendered_completed_lines,current_screen.shape[1], 3)), axis=0)
        self.screen[self.BOARD_SLICE] = current_screen

    def draw_curr_box(self, curr_piece):
        self.screen[self.CURR_SLICE] = 0
        self.draw_boxes(curr_piece, self.CURR_SLICE, self.CURRENT_PIECE_COLOR)

    def draw_next_box(self, next_piece):
        self.screen[self.NEXT_SLICE] = 0
        self.draw_boxes(next_piece, self.NEXT_SLICE, self.CURRENT_PIECE_COLOR)

    def draw_ghost_piece(self, piece, x, y):
        if self.PREV_GHOST_SLICE is not None:
            slice, pattern1 = self.PREV_GHOST_SLICE
            self.screen[slice][:pattern1.shape[0], :pattern1.shape[1]][pattern1] = 0
        pixelx, pixely = x*self.BOX_SIZE+self.EDGE_SIZE, y*self.BOX_SIZE+self.EDGE_SIZE
        slice =  np.s_[pixely:pixely+piece.shape[0]*self.BOX_SIZE,pixelx:pixelx+piece.shape[1]*self.BOX_SIZE]
        square_loc = np.repeat(np.repeat(piece, 20,axis=0), 20,axis=1)
        pattern1 = np.logical_and(np.tile(self.PATTERN_1, piece.shape), square_loc)
        self.PREV_GHOST_SLICE = (slice, pattern1)
        self.screen[slice][:pattern1.shape[0], :pattern1.shape[1]][pattern1] = self.GHOST_PIECE_COLOR

    def get_render(self):
        return self.screen

class PieceTracker:
    def __init__(self, rotations, rotation=0, x=0, y=0):
        self.rotations = rotations
        self.rotation = rotation
        self.set_var(rotations[rotation])
        self.x = x - self.cntr
        self.y = y

    def set_var(self, piece):
        self.template = piece.template
        self.cntr = piece.cntr
        self.y_dim, self.x_dim = piece.y_dim, piece.x_dim

    def set_rotation(self, rotation):
        self.rotation = rotation%len(self.rotations)
        self.set_var(self.rotations[self.rotation])

    def rotate_left(self):
        self.set_rotation(self.rotation-1)

    def rotate_right(self):
        self.set_rotation(self.rotation+1)

class GameState:
    def __init__(self, board_size=(20,10), only_squares=False):
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
        self.RENDER = DrawBoard(board_size)
        self.reset()

    def reset(self):
        self.mini_board = np.zeros((self.Y_BOARD, self.X_BOARD), dtype=bool)
        self.remaining = list(self.TETRIMINOS.keys()).copy()
        self.RENDER.reset()

        self.score = 0
        self.lines = 0
        self.combo = -1
        self.level = self.calc_level()
        self.game_over = False

        self.curr_piece = None; self.next_piece = None
        self.lock_delay = True
        self.cycle_pieces()
        self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)

    def get_new_piece(self):
        if (len(self.remaining) <= 0):
            self.remaining = list(self.TETRIMINOS.keys()).copy()
        letter = random.choice(self.remaining)
        self.remaining.remove(letter)
        piece = PieceTracker(self.TETRIMINOS[letter], x=int(self.X_BOARD/2))
        return piece

    def cycle_pieces(self):
        if self.next_piece is not None:
            self.curr_piece = self.next_piece
        else:
            self.curr_piece = self.get_new_piece()
        self.next_piece = self.get_new_piece()
        self.RENDER.draw_curr_box(self.curr_piece.template)
        self.RENDER.draw_next_box(self.next_piece.template)
        self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)
        return self.check_collision(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)

    def check_collision(self, piece, x, y):
        collisions = np.logical_and(piece, self.mini_board[y:y+piece.shape[0],x:x+piece.shape[1]])
        return collisions.any()

    def draw_piece(self, x, y=0):
        self.mini_board[y:self.curr_piece.y_dim+y, x:self.curr_piece.x_dim+x] |= self.curr_piece.template # place piece using logical or
        self.RENDER.draw_landed_piece(self.curr_piece, x, y)

    def set_piece_rotation(self, rotation):
        self.curr_piece.set_rotation(rotation)

    def set_curr_piece(self, x, force_on=True):
        x -= self.curr_piece.cntr
        if force_on: # force piece left/right onto board
            x = np.clip(x, 0, self.X_BOARD - self.curr_piece.x_dim)
        self.curr_piece.x=x

    def shift_piece_down(self, gravity=1):
        if self.curr_piece.y+self.curr_piece.y_dim < self.Y_BOARD and not self.check_collision(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y+1):
            self.curr_piece.y = min(self.Y_BOARD, self.curr_piece.y+gravity)
            self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)
            return 0, False
        else:
            self.draw_piece(self.curr_piece.x, self.curr_piece.y)
            return self.curr_piece.y-self.curr_piece.y_dim-1, True

    def rotate_piece(self, rotate):
        if rotate == 'L':
            self.curr_piece.rotate_left()
        else:
            self.curr_piece.rotate_right()
        self.RENDER.draw_curr_box(self.curr_piece.template)
        self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)
        return self.shift_piece_down()

    def move_curr_piece(self, adj_x, force_on=True):
        if force_on: # force piece left/right onto board
            self.curr_piece.x = np.clip(self.curr_piece.x+adj_x, 0, self.X_BOARD - self.curr_piece.x_dim)
        self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)
        return self.shift_piece_down()

    def hard_drop(self):
        adj_y = 0
        # starting from top, keep shifting piece down
        while adj_y+self.curr_piece.y+self.curr_piece.y_dim < self.Y_BOARD and not self.check_collision(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y+adj_y+1):
            adj_y += 1
        self.draw_piece(self.curr_piece.x, self.curr_piece.y+adj_y)
        return adj_y+self.curr_piece.y_dim-self.curr_piece.y-1, True

    def soft_drop(self):
        # starting from top, keep shifting piece down
        if self.curr_piece.y+self.curr_piece.y_dim < self.Y_BOARD and not self.check_collision(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y+1):
            self.curr_piece.y += 1
            self.RENDER.draw_ghost_piece(self.curr_piece.template, self.curr_piece.x, self.curr_piece.y)
            return self.shift_piece_down()
        else:
            self.draw_piece(self.curr_piece.x, self.curr_piece.y)
            return self.curr_piece.y-self.curr_piece.y_dim-1, True

    def remove_completed_lines(self):
        completedLines = np.all(self.mini_board, axis=1)
        num_removed_lines = sum(completedLines)
        if num_removed_lines != 0:
            # update miniboard
            self.mini_board = np.delete(self.mini_board, completedLines, axis=0)
            self.mini_board = np.insert(self.mini_board, 0, np.zeros((num_removed_lines,self.X_BOARD), dtype=bool), axis=0)
            # update rendered board
            self.RENDER.clear_rows(completedLines)
        return num_removed_lines

    def calc_level(self): # calculate level based on the lines cleared
        self.level = min(int(self.lines / 10)+1, 10)
        return self.level

    def update_score(self, piece_fallen, cleared_lines, use_score=True, scale_reward=1.0):
        additional_score = 0
        if use_score:
            self.combo = self.combo + 1 if cleared_lines > 0 else -1
            if self.combo > -1:
                additional_score += 50 * self.COMBO_MULTI[min(self.combo,self.MAX_COMBO)] * self.level # updating score for combos
            additional_score += self.LINE_MULTI[cleared_lines] * self.level  # updating score for cleared lines
            additional_score += piece_fallen # how much the piece has fallen
        additional_score += cleared_lines
        additional_score *= scale_reward
        self.score += additional_score
        return additional_score

    def get_board(self):
        screen = self.RENDER.get_render()
        return np.moveaxis(screen, -1, 0)

    def screen_dim(self):
        return (3, self.RENDER.Y_SCREEN, self.RENDER.X_SCREEN)

class TetrisEnv(gym.Env):
    def __init__(self, board_size=(20,10), action_type='grouped', output_type='image', simplified=True, score_reward=True, scale_reward=1, max_steps=10000):
        super(TetrisEnv, self).__init__()
        if simplified:
            self.state = GameState(board_size, only_squares=True)
            self.action_type = 'simplified'
            self._action_set = [a for a in range(self.state.X_BOARD)]
        else:
            self.state = GameState(board_size, only_squares=False)
            self.action_type = action_type
            if action_type == 'grouped':
                self._action_set = [a for a in range(self.state.X_BOARD * 4)]
            elif action_type == 'semigrouped':
                self._action_set = [a for a in range(self.state.X_BOARD+2)]
            elif action_type == 'standard':
                self._action_set = [a for a in range(6)]
            else:
                raise ValueError('Action type not recognised.')
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.screen_dim(), dtype=np.uint8)
        self.output_type = output_type
        self.score_reward = score_reward
        self.scale_reward = scale_reward
        self.steps = 0
        self.max_steps = max_steps
        # other stuff
        self.viewer = None

    def reset(self):
        self.state.reset()
        self.steps = 0
        state = self.state.get_board()
        return state

    def step(self, action):
        if self.action_type == 'grouped':
            x_index = int(action/4)
            rotation = action % 4
            self.state.set_piece_rotation(rotation)
            self.state.set_curr_piece(x_index, force_on=True)
            shifted_down, piece_dropped = self.state.hard_drop()
        elif self.action_type == 'semigrouped':
            if action < self.state.X_BOARD:
                self.state.set_curr_piece(action, force_on=True)
                shifted_down, piece_dropped = self.state.hard_drop()
            else:
                rotate_left = action == self.state.X_BOARD
                shifted_down, piece_dropped = self.state.rotate_piece(left=rotate_left)
        elif self.action_type == 'standard':
            # actions = {0:'rotate_left', 1: 'rotate_right', 2: 'hard_drop', 3: 'soft_drop', 4:'move_left', 5: 'move_right'}
            if action < 2:
                rotate = 'L' if action == 0 else 'R'
                shifted_down, piece_dropped = self.state.rotate_piece(rotate)
            elif action == 2:
                shifted_down, piece_dropped = self.state.hard_drop()
            elif action == 3:
                shifted_down, piece_dropped = self.state.soft_drop()
            else:
                adj_x = -1 if action == 4 else 1
                shifted_down, piece_dropped = self.state.move_curr_piece(adj_x)
        elif self.action_type == 'simplified':
            self.state.set_curr_piece(action, force_on=True)
            shifted_down, piece_dropped = self.state.hard_drop()
        if piece_dropped:
            game_over = self.state.cycle_pieces()
        else:
            game_over = False

        if not game_over:
            cleared = self.state.remove_completed_lines()
            reward = self.state.update_score(shifted_down, cleared, use_score=self.score_reward, scale_reward=self.scale_reward)
        else:
            reward = 0
        state = self.state.get_board()
        self.steps += 1
        if self.steps >= self.max_steps:
            game_over = True
        return state, reward, game_over, {}

    def render(self, mode='image', wait_sec=0, verbose=False):
        image = self.state.get_board()
        if mode =='image':
            if self.viewer is None:
                self.viewer = gym.envs.classic_control.rendering.SimpleImageViewer()
            image = np.moveaxis(image, 0, -1)
            self.viewer.imshow(image)
            time.sleep(wait_sec)
        elif mode =='rbg_array' or mode == 'human':
            return image
        elif mode == 'none':
            pass
        else:
            print(mode)
        if verbose:
            print("Curr piece:", self.state.curr_piece, ", next piece:", self.state.next_piece)
            print("Game over:", self.state.game_over)
            print(self.state.mini_board)

    def close_render(self):
        if self.viewer is not None and self.viewer.isopen:
            self.viewer.close()

    def measure_step_time(self, warmup=30, steps=1000, verbose=False):
        t = time.perf_counter()
        self.reset()
        if verbose:
            print("Warming up...")
        while (time.perf_counter()-t < warmup):
            action = random.choice(env._action_set)
            _ = env.step(action)
        if verbose:
            print("Measuring time...")
        t = time.perf_counter()
        for _ in range(steps):
            action = random.choice(env._action_set)
            _ = env.step(action)
        step_per_sec = steps/(time.perf_counter()-t)
        self.reset()
        if verbose:
            print("{} steps per second".format(step_per_sec))
        return step_per_sec

class InteractionTetris(TetrisEnv):
    def __init__(self, board_size=(20,10), action_type='grouped', only_squares=False):
        super().__init__(board_size, action_type, only_squares)

    def step(self, action):
        action = input('Type in action,'+str(self._action_set))
        action = int(action)
        print(action)
        return super().step(action)

if __name__ == '__main__':
    env = InteractionTetris(only_squares=False, action_type='standard', board_size=(10,10))
    env.reset()
    # env.measure_step_time(verbose=True)
    env.render(mode='none', wait_sec=0.1, verbose=True)
    for _ in range(100):
        action = random.choice(env._action_set)
        screen, reward, game_over, info = env.step(action)
        env.render(mode='none', wait_sec=1, verbose=True)
        print("Reward:", reward)
        if game_over:
            env.reset()
            print("Game over!")
