import numpy as np
import random
import gym
import gym.spaces as spaces
import time
import math
import torch

class TETRIMINO(object):
    templates = []
    centers = []
    type = ""

    def __init__(self, x, y=0, rotation=0):
        self.y = y
        self.x = x
        self.rotation = rotation
        self.update_parameters()

    def get_x(self):
        return self.x - self.x_cnt

    def get_y(self):
        return self.y - self.y_cnt

    def get_adj(self):
        return 2 - self.y_cnt, 2 - self.x_cnt


    def update_parameters(self):
        self.template = self.templates[self.rotation]
        self.y_dim, self.x_dim = self.template.shape
        self.y_cnt, self.x_cnt = self.centers[self.rotation]

    def set_rotation(self, rotation):
        prev_ro = self.rotation
        self.rotation = rotation % len(self.templates)
        self.update_parameters()

    def rotate(self, direction='L'):
        if direction == 'L':
            self.set_rotation(self.rotation-1)
        elif direction == 'R':
            self.set_rotation(self.rotation+1)
        else:
            raise ValueError("Direction is not L or R")

class O_TETRIMINO(TETRIMINO):
    template = np.ones((2, 2), dtype=np.int8).astype(bool)
    templates = [template]
    centers = [(1,1)]
    type = "O"

class I_TETRIMINO(TETRIMINO):
    template = np.ones((1, 4), dtype=np.int8).astype(bool)
    templates = [template, np.rot90(template)]
    centers = [(0,2),(2,0)]
    type = "I"

class S_TETRIMINO(TETRIMINO):
    template = np.zeros((2, 3), dtype=np.int8).astype(bool)
    template[0, :2] = 1
    template[1, 1:] = 1
    templates = [template, np.rot90(template)]
    centers = [(1,1),(1,0)]
    type = "S"

class Z_TETRIMINO(TETRIMINO):
    template = np.zeros((2, 3), dtype=np.int8).astype(bool)
    template[0, 1:] = 1
    template[1, :2] = 1
    templates = [template, np.rot90(template)]
    centers = [(1,1),(1,0)]
    type = "Z"

class L_TETRIMINO(TETRIMINO):
    template = np.zeros((2, 3), dtype=np.int8).astype(bool)
    template[0, 0] = 1
    template[1, :] = 1
    templates = [
        template,
        np.rot90(template, -1),
        np.rot90(template, -2),
        np.rot90(template, 1)
    ]
    centers = [(1,1),(1,0),(0,1),(1,1)]
    type = "L"

class J_TETRIMINO(TETRIMINO):
    template = np.zeros((2, 3), dtype=np.int8).astype(bool)
    template[0, 2] = 1
    template[1, :] = 1
    templates = [
        template,
        np.rot90(template, -1),
        np.rot90(template, -2),
        np.rot90(template, 1)
    ]
    centers = [(1,1),(1,0),(0,1),(1,1)]
    type = "J"

class T_TETRIMINO(TETRIMINO):
    template = np.zeros((2, 3), dtype=np.int8).astype(bool)
    template[0, 1] = 1
    template[1, :] = 1
    templates = [
        template,
        np.rot90(template, -1),
        np.rot90(template, -2),
        np.rot90(template, 1)
    ]
    centers = [(1,1),(1,0),(0,1),(1,1)]
    type = "T"

class TETRIMINO_BAG:
    def __init__(self, squares_only=False, spawn_x=3, spawn_y=0, spawn_rotation=0):
        self.squares_only = squares_only
        self.spawn_x = spawn_x
        self.spawn_y = spawn_y
        self.spawn_rotation = spawn_rotation
        self.refill_bag()

    def refill_bag(self):
        if self.squares_only:
            self.bag = [O_TETRIMINO]
        else:
            self.bag = [O_TETRIMINO, I_TETRIMINO, S_TETRIMINO, Z_TETRIMINO, L_TETRIMINO, J_TETRIMINO, T_TETRIMINO]

    def get_piece(self):
        # return I_TETRIMINO(self.spawn_x)
        if len(self.bag) < 1:
            self.refill_bag()
        return self.bag.pop(random.randint(0, len(self.bag) - 1))(self.spawn_x, self.spawn_y, self.spawn_rotation)

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
        self.Y_SCREEN  = max(board_shape[0]+self.EDGE_SIZE*2, piece_box*3+self.EDGE_SIZE*6+self.BOX_SIZE*2)
        self.X_SCREEN  = board_shape[1] + self.EDGE_SIZE * 3 + piece_box
        # colors
        self.BACKGROUND_COLOR = GRAY
        self.EMPTYBOX_COLOR = BLACK
        self.PLACED_PIECE_COLOR = DARK_GRAY
        self.CURRENT_PIECE_COLOR = GREEN
        self.GHOST_PIECE_COLOR = tuple([x*0.3 for x in self.CURRENT_PIECE_COLOR])
        # slices
        self.BOARD_SLICE = np.s_[self.EDGE_SIZE:self.EDGE_SIZE+board_shape[0],self.EDGE_SIZE:self.EDGE_SIZE+board_shape[1]]
        self.CURR_SLICE = (self.EDGE_SIZE, self.EDGE_SIZE*2+board_shape[1])
        self.NEXT_SLICE = (self.EDGE_SIZE*2+piece_box, self.EDGE_SIZE*2+board_shape[1])
        self.HOLD_SLICE = (self.EDGE_SIZE*3+piece_box*2, self.EDGE_SIZE*2+board_shape[1])
        self.LEVEL_SLICE = (self.EDGE_SIZE*4+piece_box*3,self.EDGE_SIZE*2+board_shape[1])
        self.COMBO_SLICE = (self.EDGE_SIZE*5+piece_box*3+self.BOX_SIZE,self.EDGE_SIZE*2+board_shape[1])
        # board
        self.EMPTY_BOARD = np.full((self.Y_SCREEN, self.X_SCREEN, 3), self.BACKGROUND_COLOR, dtype=np.uint8)
        self.EMPTY_BOARD[self.BOARD_SLICE] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.CURR_SLICE[0]:self.CURR_SLICE[0]+piece_box, self.CURR_SLICE[1]:self.CURR_SLICE[1]+piece_box] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.NEXT_SLICE[0]:self.NEXT_SLICE[0]+piece_box, self.NEXT_SLICE[1]:self.NEXT_SLICE[1]+piece_box] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.HOLD_SLICE[0]:self.HOLD_SLICE[0]+piece_box, self.HOLD_SLICE[1]:self.HOLD_SLICE[1]+piece_box] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.LEVEL_SLICE[0]:self.LEVEL_SLICE[0]+self.BOX_SIZE, self.LEVEL_SLICE[1]:self.LEVEL_SLICE[1]+piece_box] = self.EMPTYBOX_COLOR
        self.EMPTY_BOARD[self.COMBO_SLICE[0]:self.COMBO_SLICE[0]+self.BOX_SIZE, self.COMBO_SLICE[1]:self.COMBO_SLICE[1]+piece_box] = self.EMPTYBOX_COLOR
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

    def draw_landed_piece(self, piece, x_adj, y_adj):
        pixelx, pixely = x_adj*self.BOX_SIZE+self.EDGE_SIZE, y_adj*self.BOX_SIZE+self.EDGE_SIZE
        slice =  np.s_[pixely:pixely+piece.shape[0]*self.BOX_SIZE,pixelx:pixelx+piece.shape[1]*self.BOX_SIZE]
        self.draw_boxes(piece, slice, self.PLACED_PIECE_COLOR)
        self.PREV_GHOST_SLICE = None

    def clear_rows(self, rows):
        current_screen = self.screen[self.BOARD_SLICE]
        renderedCompletedLines = np.repeat(rows, self.BOX_SIZE, axis=0)
        num_rendered_completed_lines = sum(renderedCompletedLines)
        current_screen = np.delete(current_screen, renderedCompletedLines, axis=0)
        current_screen = np.insert(current_screen, 0, np.zeros((num_rendered_completed_lines,current_screen.shape[1], 3)), axis=0)
        self.screen[self.BOARD_SLICE] = current_screen

    def draw_curr_box(self, curr, y_adj, x_adj):
        self.screen[self.CURR_SLICE[0]:self.CURR_SLICE[0]+self.BOX_SIZE*4,self.CURR_SLICE[1]:self.CURR_SLICE[1]+self.BOX_SIZE*4] = 0
        slice = np.s_[self.CURR_SLICE[0]+y_adj*self.BOX_SIZE:self.CURR_SLICE[0]+self.BOX_SIZE*4,self.CURR_SLICE[1]+x_adj*self.BOX_SIZE:self.CURR_SLICE[1]+self.BOX_SIZE*4]
        self.draw_boxes(curr, slice, self.CURRENT_PIECE_COLOR)

    def draw_next_box(self, next, y_adj, x_adj):
        self.screen[self.NEXT_SLICE[0]:self.NEXT_SLICE[0]+self.BOX_SIZE*4,self.NEXT_SLICE[1]:self.NEXT_SLICE[1]+self.BOX_SIZE*4] = 0
        slice = np.s_[self.NEXT_SLICE[0]+y_adj*self.BOX_SIZE:self.NEXT_SLICE[0]+self.BOX_SIZE*4,self.NEXT_SLICE[1]+x_adj*self.BOX_SIZE:self.NEXT_SLICE[1]+self.BOX_SIZE*4]
        self.draw_boxes(next, slice, self.CURRENT_PIECE_COLOR)

    def draw_hold_box(self, hold, y_adj, x_adj):
        self.screen[self.HOLD_SLICE[0]:self.HOLD_SLICE[0]+self.BOX_SIZE*4,self.HOLD_SLICE[1]:self.HOLD_SLICE[1]+self.BOX_SIZE*4] = 0
        slice = np.s_[self.HOLD_SLICE[0]+y_adj*self.BOX_SIZE:self.HOLD_SLICE[0]+self.BOX_SIZE*4,self.HOLD_SLICE[1]+x_adj*self.BOX_SIZE:self.HOLD_SLICE[1]+self.BOX_SIZE*4]
        self.draw_boxes(hold, slice, self.CURRENT_PIECE_COLOR)

    def draw_combo_slice(self, combo):
        self.screen[self.COMBO_SLICE[0]:self.COMBO_SLICE[0]+self.BOX_SIZE, self.COMBO_SLICE[1]:self.COMBO_SLICE[1]+self.BOX_SIZE*4] = self.EMPTYBOX_COLOR
        self.screen[self.COMBO_SLICE[0]:self.COMBO_SLICE[0]+self.BOX_SIZE, self.COMBO_SLICE[1]:self.COMBO_SLICE[1]+int(self.BOX_SIZE*4*combo/10)] = self.CURRENT_PIECE_COLOR

    def draw_level_slice(self, level):
        self.screen[self.LEVEL_SLICE[0]:self.LEVEL_SLICE[0]+self.BOX_SIZE, self.LEVEL_SLICE[1]:self.LEVEL_SLICE[1]+self.BOX_SIZE*4] = self.EMPTYBOX_COLOR
        self.screen[self.LEVEL_SLICE[0]:self.LEVEL_SLICE[0]+self.BOX_SIZE, self.LEVEL_SLICE[1]:self.LEVEL_SLICE[1]+int(self.BOX_SIZE*4*level/10)] = self.CURRENT_PIECE_COLOR

    def undraw_ghost_piece(self):
        if self.PREV_GHOST_SLICE is not None:
            slice, pattern1 = self.PREV_GHOST_SLICE
            self.screen[slice][:pattern1.shape[0], :pattern1.shape[1]][pattern1] = 0

    def draw_ghost_piece(self, piece, x, y):
        pixelx, pixely = x*self.BOX_SIZE+self.EDGE_SIZE, y*self.BOX_SIZE+self.EDGE_SIZE
        slice =  np.s_[pixely:pixely+piece.shape[0]*self.BOX_SIZE,pixelx:pixelx+piece.shape[1]*self.BOX_SIZE]
        square_loc = np.repeat(np.repeat(piece, 20,axis=0), 20,axis=1)
        pattern1 = np.logical_and(np.tile(self.PATTERN_1, piece.shape), square_loc)
        self.PREV_GHOST_SLICE = (slice, pattern1)
        self.screen[slice][:pattern1.shape[0], :pattern1.shape[1]][pattern1] = self.CURRENT_PIECE_COLOR

    def get_render(self):
        return self.screen

class GameState:
    def __init__(self, board_size=(20,10), squares_only=False):
        # self.x_board, self.y_board = board_size
        self.Y_BOARD, self.X_BOARD = board_size[0]+2, board_size[1]
        self.LINE_MULTI = [0, 100, 300, 500, 800]
        self.COMBO_MULTI = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        self.MAX_COMBO = len(self.COMBO_MULTI)-1
        self.SOFT_DROP_SCORE = 1
        self.HARD_DROP_SCORE = 2
        self.RENDER = DrawBoard(board_size)
        self.bag = TETRIMINO_BAG(squares_only=squares_only, spawn_x=int(self.X_BOARD/2), spawn_y=2, spawn_rotation=0)
        self.reset()

    def reset(self):
        # game state
        self.mini_board = np.zeros((self.Y_BOARD, self.X_BOARD), dtype=bool)
        self.curr = None; self.next = None; self.hold = None
        self.bag.refill_bag()
        self._cycle_pieces()

        # stats
        self.score = 0
        self.lines = 0
        self.combo = -1
        self.level = self.calc_level()
        self.game_over = False
        self.can_hold_piece = True

        # rendering stuff
        self.RENDER.reset() # 2 0,    cnt 0,2         # 1,1    cnt 1,1
        y,x = self.curr.get_adj()
        self.RENDER.draw_curr_box(self.curr.template, y, x)
        y,x = self.next.get_adj()
        self.RENDER.draw_next_box(self.next.template, y, x)
        self._draw_ghost_piece()

    # this function should be called after a piece is placed to cycle the next piece
    def _cycle_pieces(self):
        self.curr = self.next if self.next else self.bag.get_piece()
        self.next = self.bag.get_piece()

    # this function should be called to check whether the current piece could be placed here
    def _check_collision(self, x, y):
        collisions = np.logical_and(self.curr.template, self.mini_board[y:y+self.curr.y_dim,x:x+self.curr.x_dim])
        return collisions.any().item()

    def _check_in_bounds(self, x, y):
        within_x = 0 <= x <= self.X_BOARD-self.curr.x_dim
        within_y = 0 <= y <= self.Y_BOARD-self.curr.y_dim
        return within_x and within_y

    def _draw_ghost_piece(self):
        if self.curr.get_y() < 2:
            self.RENDER.draw_ghost_piece(self.curr.template[2-self.curr.get_y():], self.curr.get_x(), 0)
        else:
            self.RENDER.draw_ghost_piece(self.curr.template, self.curr.get_x(), self.curr.get_y()-2)

    # this function should be called when the current piece has landed - game over when a piece is placed in the vanish zone or a piece cannot spawn
    def _land_tetris(self):
        self.mini_board[self.curr.get_y():self.curr.get_y()+self.curr.y_dim, self.curr.get_x():self.curr.get_x()+self.curr.x_dim] |= self.curr.template # place piece using logical or
        self.RENDER.undraw_ghost_piece()
        if self.curr.get_y() < 2:
            placed_in_vanish = True
            self.RENDER.draw_landed_piece(self.curr.template[2-self.curr.get_y():], self.curr.get_x(), 0)
        else:
            placed_in_vanish = False
            self.RENDER.draw_landed_piece(self.curr.template, self.curr.get_x(), self.curr.get_y()-2)
        placed_x, placed_y = self.curr.x, self.curr.y
        self._cycle_pieces()

        y,x = self.curr.get_adj()
        self.RENDER.draw_curr_box(self.curr.template, y, x)
        y,x = self.next.get_adj()
        self.RENDER.draw_next_box(self.next.template, y, x)
        self._draw_ghost_piece()
        unable_to_spawn = self._check_collision(self.curr.get_x(), self.curr.get_y())
        game_over = placed_in_vanish or unable_to_spawn
        self.can_hold_piece = True
        return game_over, (placed_x, placed_y)

    # this function should be called during an action to set the rotation
    def set_piece_rotation(self, rotation):
        difference = rotation - self.curr.rotation
        for _ in range(max(difference, -difference)):
            success = self.step_rotate_piece('R' if difference > 0 else 'L')
            if not success:
                break

    # this function should be called during an action to set the piece x index
    def set_piece_x(self, x):
        difference = x - self.curr.x
        for _ in range(max(difference, -difference)):
            success = self.step_move_piece_x('R' if difference > 0 else 'L')
            if not success:
                break

    def step_hold_action(self):
        if self.can_hold_piece:
            self.RENDER.undraw_ghost_piece()
            if self.hold:
                temp = self.hold
                self.hold = self.curr
                self.curr = temp
            else:
                self.hold = self.curr
                self._cycle_pieces()
            self.hold.__init__(int((self.X_BOARD-4)/2)) # reset held piece
            self.RENDER.draw_curr_box(self.curr.template, self.curr.y_adj, self.curr.x_adj) # TODO
            self.RENDER.draw_next_box(self.next.template, self.next.y_adj, self.next.x_adj) # TODO
            self.RENDER.draw_hold_box(self.hold.template, self.hold.y_adj, self.hold.x_adj) # TODO
            # spawn new piece at top
            self._draw_ghost_piece()
            # check if its possible
            if self.curr.y+self.curr.y_dim < 2:
                placed_in_vanish = True
            else:
                placed_in_vanish = False
            unable_to_spawn = self._check_collision(self.curr.x, self.curr.y)
            game_over = placed_in_vanish or unable_to_spawn
            self.game_over = game_over
            self.can_hold_piece = False
            return game_over
        return False

    def step_rotate_piece(self, direction):
        if self.curr.type == "O":
            if not self._check_in_bounds(self.curr.get_x(), self.curr.get_y()) or self._check_collision(self.curr.get_x(), self.curr.get_y()):
                raise ValueError('O Tetrimino failed to rotate')
            return True
        elif self.curr.type == "I":
            SRS = {
                '0>>1': [(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
                '1>>0': [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
                '1>>2': [(0,0),(-1,0),(2,0),(-1,2),(2,-1)],
                '2>>1': [(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
                '2>>3': [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
                '3>>2': [(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
                '3>>0': [(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
                '0>>3': [(0,0),(-1,0),(2,0),(-1,2),(2,-1)]
            }
        else:
            SRS = {
                '0>>1': [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
                '1>>0': [(0,0),(1,0),(1,-1),(0,2),(1,2)],
                '1>>2': [(0,0),(1,0),(1,-1),(0,2),(1,2)],
                '2>>1': [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
                '2>>3': [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
                '3>>2': [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
                '3>>0': [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
                '0>>3': [(0,0),(1,0),(1,1),(0,-2),(1,-2)]
            }
        self.RENDER.undraw_ghost_piece()
        rotation_index = str(self.curr.rotation) + '>>'
        self.curr.rotate(direction)
        rotation_index += str(self.curr.rotation)
        troubleshoot = SRS[rotation_index]
        success = False
        for x,y in troubleshoot:
            y = -y
            if self._check_in_bounds(self.curr.get_x()+x, self.curr.get_y()+y) and not self._check_collision(self.curr.get_x()+x, self.curr.get_y()+y):
                self.curr.x += x; self.curr.y += y
                success = True
                break
        if not success: # fail the rotation
            self.curr.rotate('R' if direction == 'L' else 'L')
        y,x = self.curr.get_adj()
        self.RENDER.draw_curr_box(self.curr.template, y, x)
        self._draw_ghost_piece()
        return success

    def step_move_piece_x(self, direction='L'):
        adj = -1 if direction == 'L' else 1
        if self._check_in_bounds(self.curr.get_x()+adj, self.curr.get_y()) and not self._check_collision(self.curr.get_x()+adj, self.curr.get_y()):
            self.RENDER.undraw_ghost_piece()
            self.curr.x += adj
            self._draw_ghost_piece()
            return True
        return False

    # starting from top, keep shifting piece down and then place it
    def step_hard_drop(self):
        y = 0
        while self._check_in_bounds(self.curr.get_x(), self.curr.get_y()+y+1) and not self._check_collision(self.curr.get_x(), y+self.curr.get_y()+1):
            y += 1
        self.curr.y += y
        game_over, piece_placed = self._land_tetris()
        return y, game_over, piece_placed

    # starting from top, shift piece once and then apply gravity
    def step_soft_drop(self, gravity=1):
        self.RENDER.undraw_ghost_piece()
        adj = 0
        # drop it as many blocks as possible towards the gravity
        while adj < gravity and self._check_in_bounds(self.curr.get_x(), self.curr.get_y()+adj+1) and not self._check_collision(self.curr.get_x(), adj+self.curr.get_y()+1):
            adj += 1
        self.curr.y += adj
        if adj < gravity:
            # try to land block as its clearly not able to fall the maximum height
            game_over, piece_placed = self._land_tetris()
            self._draw_ghost_piece()
            return adj, game_over, piece_placed
        else:
            self._draw_ghost_piece()
            placed_block = False
            return adj, False, None

    def remove_completed_lines(self):
        completedLines = np.all(self.mini_board, axis=1)
        num_removed_lines = sum(completedLines)
        if num_removed_lines != 0:
            # update miniboard
            self.mini_board = np.delete(self.mini_board, completedLines, axis=0)
            self.mini_board = np.insert(self.mini_board, 0, np.zeros((num_removed_lines,self.X_BOARD), dtype=bool), axis=0)
            # update rendered board
            self.RENDER.undraw_ghost_piece()
            self.RENDER.clear_rows(completedLines[2:])
            self._draw_ghost_piece()
        return num_removed_lines

    def calc_level(self): # calculate level based on the lines cleared
        self.level = min(int(self.lines / 10)+1, 10)
        return self.level

    def update_score(self, piece_fallen, cleared_lines, reward_type='standard', reward_scaling=None, hard_drop=False, soft_drop=False):
        additional_score = 0
        level = self.level
        self.combo = self.combo + 1 if cleared_lines > 0 else -1
        self.RENDER.draw_level_slice(self.level)
        self.RENDER.draw_combo_slice(self.combo)
        if reward_type == 'standard':
            additional_score += self.LINE_MULTI[cleared_lines] * level  # updating score for cleared lines
            if self.combo > -1:
                additional_score += 50 * self.COMBO_MULTI[min(self.combo,self.MAX_COMBO)] * level # updating score for combos
            if soft_drop:
                additional_score += piece_fallen * self.SOFT_DROP_SCORE # how much the piece has fallen
            if hard_drop:
                additional_score += piece_fallen * self.HARD_DROP_SCORE
            additional_score += cleared_lines
        elif reward_type == 'no piece drop':
            additional_score += self.LINE_MULTI[cleared_lines] * level  # updating score for cleared lines
            if self.combo > -1:
                additional_score += 50 * self.COMBO_MULTI[min(self.combo,self.MAX_COMBO)] * level # updating score for combos
            additional_score += cleared_lines
        elif reward_type == 'only lines':
            additional_score += self.LINE_MULTI[cleared_lines] * level  # updating score for cleared lines
        elif reward_type == 'num lines':
            additional_score += cleared_lines
        else:
            raise ValueError('Reward type not recognised.')
        if reward_scaling is not None:
            if reward_scaling == 'multi':
                additional_score *= 0.001
            elif reward_scaling == 'log':
                additional_score = math.log10(additional_scale)
        self.score += additional_score
        return float(additional_score)

    def get_board(self, scale=False):
        screen = self.RENDER.get_render()
        return np.moveaxis(screen, -1, 0)

    def screen_dim(self):
        return (3, self.RENDER.Y_SCREEN, self.RENDER.X_SCREEN)

class TetrisEnv(gym.Env):
    def __init__(self, board_size=(20,10), action_type='grouped', reward_type='standard', reward_scaling=None, max_steps=None):
        super(TetrisEnv, self).__init__()
        self.action_type = action_type
        self.state = GameState(board_size, squares_only=True) if action_type == 'simplified' else GameState(board_size, squares_only=False)
        if action_type == 'simplified':
            self._action_set = [a for a in range(self.state.X_BOARD)]
        elif action_type == 'grouped':
            self._action_set = [a for a in range(self.state.X_BOARD * 4)]
        elif action_type == 'semigrouped':
            self._action_set = [a for a in range(self.state.X_BOARD+2)]
        elif action_type == 'standard':
            self._action_set = [a for a in range(6)]
        else:
            raise ValueError('Action type not recognised.')
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.state.screen_dim(), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=0, high=1e3, shape=(1,62))
        self.reward_type = reward_type
        self.reward_scaling = reward_scaling
        self.steps = 0
        self.max_steps = max_steps
        # other stuff
        self.viewer = None
        self.frame_buffer = np.zeros(self.state.screen_dim(), 'float32')

    def update_frame_buffer(self, img):
        self.frame_buffer = np.concatenate([img, self.frame_buffer[:-3]], axis = 0)

    def reset(self):
        self.framebuffer = np.zeros(self.state.screen_dim(), 'float32')
        self.state.reset()
        self.steps = 0
        state = self.state.get_board(scale=True)
        return state

    def step(self, action):
        info = {}
        soft_drop = False
        hard_drop = False
        if self.action_type == 'grouped':
            x_index = int(action/4); rotation = action % 4
            self.state.set_piece_rotation(rotation)
            self.state.set_piece_x(x_index)
            shifted_down, game_over, landed_tetris = self.state.step_hard_drop()
            hard_drop = True
        elif self.action_type == 'semigrouped':
            if action < self.state.X_BOARD:
                self.state.set_piece_x(action)
                shifted_down, game_over, landed_tetris = self.state.step_hard_drop()
                hard_drop = True
            else:
                rotate = 'L' if action == self.state.X_BOARD else 'R'
                self.state.step_rotate_piece(rotate)
                shifted_down, game_over, landed_tetris = self.state.step_soft_drop()
        elif self.action_type == 'standard':
            # actions = {0:'rotate_left', 1: 'rotate_right', 2: 'hard_drop', 3: 'soft_drop', 4:'move_left', 5: 'move_right'}
            if action < 2:
                rotate = 'L' if action == 0 else 'R'
                self.state.step_rotate_piece(rotate)
                shifted_down, game_over, landed_tetris = self.state.step_soft_drop()
            elif action == 2:
                shifted_down, game_over, landed_tetris = self.state.step_hard_drop()
                hard_drop = True
            elif action == 3:
                shifted_down, game_over, landed_tetris = self.state.step_soft_drop(gravity=2)
                soft_drop = True
            else:
                x_adj = 'L' if action == 4 else 'R'
                self.state.step_move_piece_x(x_adj)
                shifted_down, game_over, landed_tetris = self.state.step_soft_drop()
        elif self.action_type == 'simplified':
            self.state.set_piece_x(action)
            shifted_down, game_over, landed_tetris = self.state.step_hard_drop()
        cleared = self.state.remove_completed_lines()
        info['lines cleared'] = cleared
        info['block placed'] = landed_tetris
        info['level'] = self.state.level
        info['combo'] = self.state.combo
        reward = self.state.update_score(shifted_down, cleared, reward_type=self.reward_type, reward_scaling=self.reward_scaling, hard_drop=hard_drop, soft_drop=soft_drop)
        next_state = self.state.get_board(scale=True)
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps:
            game_over = True
        return next_state, reward, game_over, info

    def render(self, mode='image', wait_sec=0, verbose=False):
        image = self.state.get_board()
        if mode =='image':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
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
            print("Curr piece:", self.state.curr, ", next piece:", self.state.next)
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
            _, _, game_over, _ = env.step(action)
            if game_over:
                env.reset()
        if verbose:
            print("Measuring time...")
        t = time.perf_counter()
        for _ in range(steps):
            action = random.choice(env._action_set)
            _, _, game_over, _ = env.step(action)
            if game_over:
                env.reset()
        step_per_sec = steps/(time.perf_counter()-t)
        self.reset()
        if verbose:
            print("{} steps per second".format(step_per_sec))
        return step_per_sec

class StackedFrameTetris():
    pass

class InteractionTetris(TetrisEnv):
    def __init__(self, board_size=(20,10), action_type='grouped'):
        super().__init__(board_size, action_type)

    def step(self, action):
        if self.action_type == 'grouped':
            string = f'Type in index [0-{self.state.X_BOARD-1}] and rotation [0-4] in format \"index,rotation\":'
            action = input(string).split(',')
            action = int(action[0]) * 4 + int(action[1])
        elif self.action_type == 'semigrouped':
            string = f'Type in index [0-{self.state.X_BOARD-1}] or rotation [L,R]:'
            action = input(string)
            if action.isnumeric():
                action = int(action)
            elif action == 'L':
                action = self.state.X_BOARD
            else:
                action = self.state.X_BOARD+1
        elif self.action_type == 'standard':
            string = f'Type in action [0: rotate left, 1: rotate right, 2: hard drop, 3: soft drop, 4: move left, 5: move right]:'
            #{0:'rotate_left', 1: 'rotate_right', 2: 'hard_drop', 3: 'soft_drop', 4:'move_left', 5: 'move_right'}
            action = int(input(string))
        else:
            action = int(input('Type in action (index) '+ str(self._action_set)))
        return super().step(action)

if __name__ == '__main__':
    env = InteractionTetris(action_type='semigrouped', board_size=(20,10))
    # env.measure_step_time(verbose=True)
    env.render(mode='image', wait_sec=0.1, verbose=True)
    for _ in range(100):
        action = random.choice(env._action_set)
        screen, reward, game_over, info = env.step(action)
        env.render(mode='image', wait_sec=1, verbose=True)
        print("Reward:", reward)
        if game_over:
            env.reset()
            print("Game over!")
