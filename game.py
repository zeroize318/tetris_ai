import sys

import pygame

from lib import helper
from tetromino import Tetromino
import random
import numpy as np
from gui import Gui
import time
from common import *

INITIAL_EX_WIGHT = 0.0
SPIN_SHIFT_FOR_NON_T = [(1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (1, 1, 0), (-1, 1, 0),
                        (1, -1, 0), (-1, -1, 0),
                        (0, 2, 0), (1, 2, 0), (-1, 2, 0)]

# if you don't want to see some spurious t-spin moves
SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (1, 1, 0), (-1, 1, 0),
                    (1, -1, 0), (-1, -1, 0),
                    (0, 2, 0), (1, 2, 0), (-1, 2, 0)]  # disable triple t-spin

# if you allow t spins in various funky ways
# SPIN_SHIFT_FOR_T = [(1, 0, 0), (-1, 0, 0),
#                     (0, 1, 0), (1, 1, 0), (-1, 1, 0),
#                     (0, 2, 0), (1, 2, 0), (-1, 2, 0),
#                     (0, -1, 0), (1, -1, 0), (-1, -1, 0)]  # enable triple t-spin

ACTIONS = [
    "left", "right", "down", "turn left", "turn right", "drop"
]

IDLE_MAX = 9999


class Gamestate:
    def __init__(self, grid=None, seed=None, rd=None, height=0):
        if seed is None:
            self.seed = random.randint(0, round(9e9))
        else:
            self.seed = seed
        self.rand_count = 0

        if rd is None:
            self.rd = random.Random(seed)
        else:
            self.rd = rd

        if grid is None:
            self.grid = self.initial_grid(height)
        else:
            self.grid = list()
            for row in grid:
                self.grid.append(list(row))

        self.tetromino = Tetromino.new_tetromino_fl(self.get_random().random())
        self.hold_type = None
        self.next = list()
        for i in range(5):
            self.next.append(Tetromino.random_type_str(self.get_random().random()))
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.n_lines = [0, 0, 0, 0]
        self.t_spins = [0, 0, 0, 0]
        self.game_status = "playing"
        self.is_hold_last = False
        self.ex_weight = INITIAL_EX_WIGHT
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.idle = 0
        self.combo = 0

    def start(self):
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())

    def initial_grid(self, height=0):
        grid = list()
        for _ in range(GAME_BOARD_HEIGHT):
            grid.append([0] * GAME_BOARD_WIDTH)

        if height == 0: return grid

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, Tetromino.pool_size())
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        # for j in range(GAME_BOARD_WIDTH):
        #     grid[GAME_BOARD_HEIGHT - height][j] = self.get_random().randint(0, 1)

        return grid

    def get_random_grid(self):
        grid = list()
        for i in range(GAME_BOARD_HEIGHT):
            row = list()
            for j in range(GAME_BOARD_WIDTH):
                row.append(0)
            grid.append(row)

        height = self.get_random().randint(0, min(15, GAME_BOARD_HEIGHT - 2))

        # if height = 15, range(6, 20), saving the first row for random generation
        for i in range(GAME_BOARD_HEIGHT - height, GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                grid[i][j] = self.get_random().randint(0, len(Tetromino.pool_size()))
            grid[i][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        all_brick = True
        for j in range(GAME_BOARD_WIDTH):
            grid[GAME_BOARD_HEIGHT - height - 1][j] = self.get_random().randint(0, 1)
            if grid[GAME_BOARD_HEIGHT - height - 1][j] == 0: all_brick = False
        if all_brick: grid[GAME_BOARD_HEIGHT - height - 1][self.get_random().randint(0, GAME_BOARD_WIDTH - 1)] = 0

        return grid

    @staticmethod
    def random_gamestate(seed=None):
        if seed is None:
            large_int = 999999999
            seed = random.randint(0, large_int)
        gamestate = Gamestate(seed=seed)
        gamestate.grid = gamestate.get_random_grid()
        return gamestate

    def copy(self):
        state_copy = Gamestate(self.grid, rd=self.rd)

        state_copy.seed = self.seed
        state_copy.tetromino = self.tetromino.copy()
        state_copy.hold_type = self.hold_type
        state_copy.next = list()
        for s in self.next:
            state_copy.next.append(s)
        state_copy.next_next = self.next_next
        state_copy.n_lines = list(self.n_lines)
        state_copy.t_spins = list(self.t_spins)
        state_copy.game_status = self.game_status
        state_copy.is_hold_last = self.is_hold_last
        state_copy.ex_weight = self.ex_weight
        state_copy.score = self.score
        state_copy.lines = self.lines
        state_copy.pieces = self.pieces
        state_copy.rand_count = self.rand_count
        state_copy.idle = self.idle
        state_copy.combo = self.combo

        return state_copy

    def copy_value(self, state_original):
        for i in range(GAME_BOARD_HEIGHT):
            for j in range(GAME_BOARD_WIDTH):
                self.grid[i][j] = state_original.grid[i][j]

        self.seed = state_original.seed
        self.tetromino = state_original.tetromino.copy()
        self.hold_type = state_original.hold_type
        for i in range(len(self.next)):
            self.next[i] = state_original.next[i]
        self.next_next = state_original.next_next
        self.n_lines = list(state_original.n_lines)
        self.t_spins = list(state_original.t_spins)
        self.game_status = state_original.game_status
        self.is_hold_last = state_original.is_hold_last
        self.ex_weight = state_original.ex_weight
        self.score = state_original.score
        self.lines = state_original.lines
        self.pieces = state_original.pieces
        self.rand_count = state_original.rand_count
        self.idle = state_original.idle
        self.combo = state_original.combo

    def put_tet_to_grid(self, tetro=None):
        grid_copy = helper.copy_2d(self.grid)
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x > GAME_BOARD_WIDTH or y > GAME_BOARD_HEIGHT:
                continue
            if y < 0:
                continue
            grid_copy[y][x] = tetro.to_num()
        return grid_copy

    def check_collision(self, tetro=None):
        if tetro is None:
            tetro = self.tetromino

        disp = tetro.get_displaced()
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if x < 0 or x >= GAME_BOARD_WIDTH or y >= GAME_BOARD_HEIGHT:
                return True
            if y < 0:
                continue
            if self.grid[y][x] != 0:
                return True
        return False

    def check_t_spin(self):
        if self.tetromino.type_str != "T" or self.tetromino.rot != 2: return False
        check_mov = [(0, -1, 0),
                     (1, 0, 0),
                     (-1, 0, 0)]
        for mov in check_mov:
            tetro = self.tetromino.copy().move(mov)
            if not self.check_collision(tetro): return False

        return True

    def check_completed_lines(self, above_grid=None):
        completed_lines = 0
        row_num = 0
        for row in self.grid:
            complete = True
            for sq in row:
                if sq == 0:
                    complete = False
                    break
            if complete:
                self.remove_line(row_num, above_grid=above_grid)
                completed_lines += 1
            row_num += 1

        return completed_lines

    def remove_line(self, row_num, above_grid=None):
        self.grid[1:row_num + 1] = self.grid[:row_num]
        if above_grid is None:
            new_row = [0] * GAME_BOARD_WIDTH
        else:
            new_row = above_grid[:]
        self.grid[0] = new_row

    def check_clear_board(self):
        for i in reversed(range(GAME_BOARD_HEIGHT)):
            for block in self.grid[i]:
                if block != 0:
                    return False
        return True

    def update_score(self, lines, is_t_spin, is_clear, combo):
        if is_t_spin:
            if lines == 1:
                score_lines = 2
            elif lines == 2:
                score_lines = 4
            elif lines == 3:
                score_lines = 5
            else:
                score_lines = 0
            self.t_spins[lines] += 1
        else:
            score_lines = lines

        add_score = (score_lines + 1) * score_lines / 2 * 10

        if is_clear:
            add_score += 60

        if T_SPIN_MARK and is_t_spin:
            self.score = int(self.score) + add_score + 0.1
            add_score += 0.1
        else:
            self.score += add_score
        self.lines += lines

        if add_score != 0:
            if 1 < combo <= 3:
                self.score += 10
            elif 3 < combo <= 5:
                self.score += 20
            elif 5 < combo <= 8:
                self.score += 30
            elif combo > 8:
                self.score += 40

        if lines != 0: self.n_lines[lines - 1] += 1
        self.pieces += 1
        return add_score

    def get_score_text(self):
        s = "score:  " + str(int(self.score)) + "\n"
        s += "lines:  " + str(int(self.lines)) + "\n"
        s += "pieces: " + str(self.pieces) + "\n"
        one_line = ''
        for num in self.n_lines:
            one_line += f'{num} '
        s += "n_lines: " + one_line + '\n'
        one_line = ''
        for num in self.t_spins:
            one_line += f'{num} '
        s += "t_spins: " + one_line + '\n'
        s += "combo: " + f'{self.combo}\n'
        return s

    def get_info_text(self):
        # s = "unfinished info text \n"
        s = "seed: " + str(self.seed)
        return s

    def soft_drop(self):
        tetro = self.tetromino
        down = 0
        while not self.check_collision(tetro.move((0, 1, 0))): down += 1
        tetro.move((0, -1, 0))
        return down

    def hard_drop(self):
        self.soft_drop()
        return self.process_down_collision()

    def process_down_collision(self):
        is_t_spin = self.check_t_spin()
        is_above_grid = self.tetromino.check_above_grid()
        above_grid = self.tetromino.to_above_grid()
        self.freeze()
        completed_lines = self.check_completed_lines(above_grid=above_grid)
        is_clear = self.check_clear_board()
        add_score = self.update_score(completed_lines, is_t_spin, is_clear, self.combo)
        if add_score == 0:
            self.combo = 0
        else:
            self.combo += 1

        if self.check_collision() or (is_above_grid and completed_lines == 0):
            self.game_status = "gameover"
            done = True
        else:
            done = False
        return add_score, done

    def process_turn(self):  # return true if turn is successful
        if self.check_collision():
            success = False
            shifted = None
            if self.tetromino.type_str.lower() == 't':
                spin_moves = SPIN_SHIFT_FOR_T
            else:
                spin_moves = SPIN_SHIFT_FOR_NON_T
            for mov in spin_moves:
                shifted = self.tetromino.copy().move(mov)
                if not self.check_collision(shifted):
                    success = True
                    break
            if success:
                self.tetromino = shifted
            return success
        else:
            return True

    def process_left_right(self):
        if self.check_collision():
            return False
        else:
            return True

    def check_equal(self, gamestate):
        if self.is_hold_last != gamestate.is_hold_last or self.hold_type != gamestate.hold_type:
            return False
        if self.tetromino.type_str != gamestate.tetromino.type_str:
            return False
        for i in range(4):
            if self.next[i] != gamestate.next[i]:
                return False
        for r in range(GAME_BOARD_HEIGHT):
            for c in range(GAME_BOARD_WIDTH):
                if self.grid[r][c] != gamestate.grid[r][c]:
                    return False
        return True

    @classmethod
    def cls_put_tet_to_grid(cls, grid, tetro):
        grid_copy = helper.copy_2d(grid)
        disp = tetro.get_displaced()
        collide = False
        for sq in disp:
            x = sq[0]
            y = sq[1]
            if grid_copy[y][x] != 0:
                collide = True
            grid_copy[y][x] = tetro.to_num()
        return grid_copy, collide

    def hold(self):
        if self.is_hold_last: return False

        new_hold_type = self.tetromino.type_str
        if self.hold_type is None:
            self.tetromino = Tetromino.new_tetromino(self.next[0])
            self.next[:-1] = self.next[1:]
            self.next[-1] = self.next_next
            self.next_next = Tetromino.random_type_str(self.get_random().random())
        else:
            self.tetromino = Tetromino.new_tetromino(self.hold_type)

        self.hold_type = new_hold_type
        self.is_hold_last = True
        self.pieces += 1

        if self.check_collision():
            self.game_status = "gameover"
        return True

    def freeze(self):
        self.grid = self.put_tet_to_grid()
        self.tetromino = Tetromino.new_tetromino(self.next[0])
        self.next[:-1] = self.next[1:]
        self.next[-1] = self.next_next
        self.next_next = Tetromino.random_type_str(self.get_random().random())
        self.is_hold_last = False

    def get_random(self):
        # self.rand_count += 1
        # return random.Random(self.rand_count * self.seed)
        return self.rd

    def check_up_collision(self):
        self.tetromino.move((0, -1, 0))
        collision = self.check_collision()
        self.tetromino.move((0, 1, 0))
        return collision

    def get_turn_expansion(self):
        state_turn = self.copy()
        states_turn = [state_turn.copy()]
        moves_turn = [[]]
        for i in range(1, self.tetromino.rot_max):
            state_turn.tetromino.move((0, 0, 1))
            success = state_turn.process_turn()
            if not success:
                # usually not a concern until the end when
                # there is a slight chance that you cannot turn
                break
            state = state_turn.copy()
            states_turn += [state]
            moves_turn += [["turn left"] * i]

        return states_turn, moves_turn

    def get_left_right_expansion(self, moves_turn):
        # move 0
        states_lr = [self.copy()]
        moves_lr = [moves_turn]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision():
                break
            else:
                left += 1
                moves = moves_turn + ["left"] * left
                states_lr += [state_copy.copy()]
                moves_lr += [moves]

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision():
                break
            else:
                right += 1
                moves = moves_turn + ["right"] * right
                states_lr += [state_copy.copy()]
                moves_lr += [moves]

        # soft drop
        for s, m in list(zip(states_lr, moves_lr)):
            s.soft_drop()
            m += ["soft"]

        return states_lr, moves_lr

    def get_tuck_spin_expansion(self, moves_lr):
        # move 0
        states_ts = [self.copy()]
        moves_ts = [moves_lr]

        # move left
        state_copy = self.copy()
        left = 0
        while True:
            state_copy.tetromino.move((-1, 0, 0))
            if state_copy.check_collision():
                break
            elif not state_copy.check_up_collision():
                break
            else:
                left += 1
                moves = moves_lr + ["left"] * left
                states_ts += [state_copy.copy()]
                moves_ts += [moves]

        # move right
        state_copy = self.copy()
        right = 0
        while True:
            state_copy.tetromino.move((1, 0, 0))
            if state_copy.check_collision():
                break
            elif not state_copy.check_up_collision():
                break
            else:
                right += 1
                moves = moves_lr + ["right"] * right
                states_ts += [state_copy.copy()]
                moves_ts += [moves]

        if self.tetromino.rot_max == 1:
            return states_ts, moves_ts

        more_states_ts = list()
        more_moves_ts = list()
        for i in range(len(states_ts)):
            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, 1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts += [state_copy]
                more_moves_ts.append(moves_ts[i] + ["turn left"] * 1)

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, 1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts += [state_copy]
                        more_moves_ts.append(moves_ts[i] + ["turn left"] * 2)

            if self.tetromino.rot_max == 2:
                continue

            state_copy = states_ts[i].copy()
            state_copy.tetromino.move((0, 0, -1))
            if state_copy.process_turn() and state_copy.check_up_collision():
                more_states_ts += [state_copy]
                more_moves_ts.append(moves_ts[i] + ["turn right"] * 1)

                if self.tetromino.rot_max > 2:
                    state_copy = state_copy.copy()
                    state_copy.tetromino.move((0, 0, -1))
                    if state_copy.process_turn() and state_copy.check_up_collision():
                        more_states_ts += [state_copy]
                        more_moves_ts.append(moves_ts[i] + ["turn right"] * 2)

        return states_ts + more_states_ts, moves_ts + more_moves_ts

    def get_height_sum(self):
        heights = self.get_heights()
        return sum(heights)

    def get_hole_depth(self):
        depth = [0] * GAME_BOARD_WIDTH
        highest_brick = 0
        for j in range(GAME_BOARD_WIDTH):
            has_found_brick = False
            for i in range(GAME_BOARD_HEIGHT):
                if not has_found_brick:
                    if self.grid[i][j] > 0:
                        has_found_brick = True
                        highest_brick = i
                elif self.grid[i][j] == 0:
                    depth[j] = i - highest_brick
                    break
        return depth

    def get_heights(self):
        heights = [0] * GAME_BOARD_WIDTH
        for j in range(GAME_BOARD_WIDTH):
            for i in range(GAME_BOARD_HEIGHT):
                if self.grid[i][j] > 0:
                    heights[j] = GAME_BOARD_HEIGHT - i
                    break
        return heights


class Game:
    def __init__(self, gui=None, seed=None, height=0):
        self.gui = gui
        self.seed = seed
        self.current_state = Gamestate(seed=seed, height=height)
        self.all_possible_states = []
        self.height = height

    def act(self, action):
        if self.current_state.game_status == "gameover":
            return self.get_state_input(self.current_state), 0, True, False

        success = False
        done = False
        add_score = 0
        action = action.lower()

        copy_state = self.current_state.copy()

        if action == "left":
            copy_state.tetromino.move((-1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "right":
            copy_state.tetromino.move((1, 0, 0))
            success = copy_state.process_left_right()
        elif action == "turn left":
            copy_state.tetromino.move((0, 0, 1))
            success = copy_state.process_turn()
        elif action == "turn right":
            copy_state.tetromino.move((0, 0, -1))
            success = copy_state.process_turn()
        elif action == "down":
            copy_state.tetromino.move((0, 1, 0))
            if copy_state.check_collision():
                copy_state.tetromino.move((0, -1, 0))
                add_score, done = copy_state.process_down_collision()
            success = True  # move down will take effect no matter what
        elif action == "soft":
            # not a real move for human players
            copy_state.soft_drop()
            success = True
        elif action == "drop":
            add_score, done = copy_state.hard_drop()
            success = True
        elif action == "hold":
            success = copy_state.hold()
        else:
            print(str(action) + " action is not found. Please check.")

        if success:
            self.current_state = copy_state

        if action == "down" or action == "drop" or action == "hold":
            self.current_state.idle = 0
        elif self.current_state.idle >= IDLE_MAX:
            self.current_state.idle = 0
            self.current_state.tetromino.move((0, 1, 0))
            if self.current_state.check_collision():
                self.current_state.tetromino.move((0, -1, 0))
                add_score, done = self.current_state.process_down_collision()
            success = True  # move down will take effect no matter what
        else:
            self.current_state.idle += 1

        return self.get_state_input(self.current_state), add_score, done, success

    def render(self):
        if self.gui is not None:
            self.update_gui()
            self.gui.redraw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

    def restart(self, height=None):
        if height is None:
            self.current_state = Gamestate(seed=self.seed, height=self.height)
        else:
            self.current_state = Gamestate(seed=self.seed, height=height)
        self.current_state.start()

    def update_gui(self, gamestate=None, is_display_current=True):
        if self.gui is None: return
        if gamestate is None:
            gamestate = self.current_state

        if is_display_current:
            above_grid = gamestate.tetromino.to_above_grid()
            main_grid = helper.copy_2d(gamestate.put_tet_to_grid())
        else:
            above_grid = [0] * GAME_BOARD_WIDTH
            main_grid = helper.copy_2d(gamestate.grid)

        hold_grid = Tetromino.to_small_window(gamestate.hold_type)
        next_grids = list()
        for n in gamestate.next:
            next_grids.append(Tetromino.to_small_window(n))
        self.gui.update_grids_color((main_grid, hold_grid, next_grids), above_grid)

        self.gui.set_score_text(gamestate.get_score_text())
        self.gui.set_info_text(gamestate.get_info_text())

    def run(self):
        is_run = True
        while is_run:
            if self.gui is not None:
                self.update_gui()
                self.gui.redraw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.act("left")
                    if event.key == pygame.K_d:
                        self.act("right")
                    if event.key == pygame.K_s:
                        self.act("down")
                    if event.key == pygame.K_j:
                        self.act("turn left")
                    if event.key == pygame.K_k:
                        self.act("turn right")
                    if event.key == pygame.K_SPACE:
                        self.act("drop")
                    if event.key == pygame.K_q:
                        self.act("hold")
                    if event.key == pygame.K_r:
                        self.current_state = Gamestate(seed=self.seed)
                        self.current_state.start()
                    if event.key == pygame.K_1:
                        self.display_all_possible_state()
                    if event.key == pygame.K_i:
                        self.info_print()
                    if event.key == pygame.K_2:
                        # changing current tetromino
                        pool_size = Tetromino.pool_size()
                        num = self.current_state.tetromino.to_num()
                        # remember the return num has already been increased by 1, leaving room for 0
                        if num >= pool_size:
                            num = num - pool_size
                        self.current_state.tetromino = Tetromino.new_tetromino_num(num)

    def info_print(self):
        print(self.current_state.score)

        return None

    def reset(self, height=None):
        if height is None:
            self.restart()
        else:
            self.restart(height=height)
        return self.get_state_input(self.current_state)

    def step(self, action=None, chosen=None):
        if action is not None:
            return self.act(action)
        elif chosen is not None:
            self.current_state = self.all_possible_states[chosen]
            return self.get_state_input(self.current_state)
        else:
            print('something is wrong with the args in step()')
            return None

    def is_done(self):
        if self.current_state.game_status == 'gameover':
            return True
        else:
            return False

    @staticmethod
    def get_state_input(gamestate):
        if STATE_INPUT == 'long' or STATE_INPUT == 'short':
            input_ = np.concatenate([np.reshape(Game.get_main_grid_input(gamestate), [1, -1]),
                                     Game.get_height_hole_hold_next_input(gamestate)], axis=1)
        elif STATE_INPUT == 'dense':
            input_ = Game.get_height_hole_hold_next_input(gamestate)
        else:
            input_ = None
            sys.stderr('STATE_INPUT is wrong. Exit...')
            exit()

        return input_

    @staticmethod
    def get_main_grid_input(gamestate):
        buffer = []
        for i in range(len(gamestate.grid)):
            for j in range(len(gamestate.grid[i])):
                buffer.append([gamestate.grid[i][j]])

        buffer = np.reshape(np.array(buffer), [1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1])
        buffer = (buffer > 0)
        return buffer

    @staticmethod
    def get_hole_np_dqn(gamestate):
        buffer = gamestate.get_hole_depth() + gamestate.get_heights()
        return np.reshape(np.array(buffer), [1, GAME_BOARD_WIDTH * 2, 1])

    @staticmethod
    def get_height_hole_hold_next_input(gamestate):
        # part1: heights + hold_depth. len -> 20
        if STATE_INPUT == 'short':
            buffer1 = [sum(gamestate.get_heights())] + [sum(gamestate.get_hole_depth())] + [gamestate.combo]
        elif STATE_INPUT == 'long' or STATE_INPUT == 'dense':
            buffer1 = gamestate.get_heights() + gamestate.get_hole_depth() + [gamestate.combo]
        else:
            buffer1 = None
            sys.stdout.write('STATE_INPUT is wrong. Exit...')
            exit()

        # part2: current 1; hold 1; next 4
        # next will always be the last for convenience, because of the change in the last one
        # hold has one more position to record if last step is 'hold'
        hold_num = 1
        current_num = 1
        next_num = 4
        pool_size = Tetromino.pool_size()
        buffer2 = [0] * (pool_size * (hold_num + current_num + next_num) + hold_num)

        if hold_num == 1:
            if gamestate.is_hold_last:
                buffer2[0] = 1
            if gamestate.hold_type is not None:
                tetro_type_num = Tetromino.type_str_to_num(gamestate.hold_type) - 1
                buffer2[tetro_type_num + hold_num] = 1

        tetro_type_num = Tetromino.type_str_to_num(gamestate.tetromino.type_str) - 1
        buffer2[hold_num + hold_num * pool_size + tetro_type_num] = 1

        for i in range(next_num):
            tetro_type_num = Tetromino.type_str_to_num(gamestate.next[i]) - 1
            buffer2[hold_num + (i + hold_num + current_num) * pool_size + tetro_type_num] = 1

        return np.reshape(np.array(buffer1 + buffer2, dtype='int8'), [1, -1])

    def get_all_possible_gamestates(self, gamestate=None):
        if gamestate is None:
            gamestate_original = self.current_state.copy()
        else:
            gamestate_original = gamestate.copy()

        if gamestate_original.game_status == 'gameover':
            return [gamestate_original], [], [0], [True], [False], [False]

        states_lr_all = list()
        moves_lr_all = list()
        ss, ms = gamestate_original.get_turn_expansion()
        for s, m in list(zip(ss, ms)):
            s_lr, m_lr = s.get_left_right_expansion(m)
            states_lr_all += s_lr
            moves_lr_all += m_lr

        gamestates = list()
        moves = list()
        for s, m in list(zip(states_lr_all, moves_lr_all)):
            s_ts, m_ts = s.get_tuck_spin_expansion(m)
            gamestates += s_ts
            moves += m_ts

        add_scores = list()
        dones = list()

        # press down
        for s, m in list(zip(gamestates, moves)):
            add_score, done = s.hard_drop()
            m += ["drop"]
            add_scores += [add_score]
            dones += [done]

        is_include_hold = False
        is_new_hold = False
        # hold
        if gamestate_original.hold_type != gamestate_original.tetromino.type_str and \
                not gamestate_original.is_hold_last:
            is_include_hold = True
            if gamestate_original.hold_type is None:
                is_new_hold = True
            gamestate_original.hold()
            gamestates += [gamestate_original.copy()]
            moves += [["hold"]]
            add_scores += [0]
            if gamestate_original.game_status == "gameover":
                dones += [True]
            else:
                dones += [False]

        # gamestate is for GameMini; state is for neural network
        self.all_possible_states = gamestates

        return gamestates, moves, add_scores, dones, is_include_hold, is_new_hold

    def get_all_possible_states_input(self, original_gamestate=None):
        if original_gamestate is None:
            gamestates, moves, add_scores, dones, is_include_hold, is_new_hold = self.get_all_possible_gamestates(
                self.current_state)
        else:
            gamestates, moves, add_scores, dones, is_include_hold, is_new_hold = self.get_all_possible_gamestates(
                original_gamestate)

        state_input = list()
        for gamestate in gamestates:
            state_input.append(Game.get_state_input(gamestate))

        return np.concatenate(state_input), np.array([add_scores]).reshape(
            [-1, 1]), dones, is_include_hold, is_new_hold, moves, gamestates

    def display_all_possible_state(self):
        if self.gui is None: return

        states, moves, _, _, _, _ = self.get_all_possible_gamestates()
        for s, m in zip(states, moves):
            self.update_gui(s, is_display_current=False)
            self.gui.set_info_text(helper.text_list_flatten(m))
            self.gui.redraw()
            time.sleep(0.1)

    def get_moves(self, target_gamestate, current_gamestate=None):
        if current_gamestate is None:
            current_gamestate = self.current_state

        all_possible_gamestates, moves, _, _, _, _ = self.get_all_possible_gamestates(current_gamestate)
        for i in range(len(all_possible_gamestates)):
            if target_gamestate.check_equal(all_possible_gamestates[i]):
                return moves[i]

        sys.stderr('WARNING: cannot find the moves from current gamestate to target gamestate.')
        return []


if __name__ == "__main__":
    game = Game(gui=Gui(), seed=None)
    game.restart()
    game.run()
