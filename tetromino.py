import random
from common import *


class Tetromino:
    MASSIVE_WEIGHT = 0.135  # the chance of a piece whose name ends with 'massive' compared with other pieces
    RNG_THRESHOLD = list()
    __POOL = list()

    @classmethod
    def create_pool(cls):
        if len(cls.__POOL) != 0: return
        # regular Tetris
        if GAME_TYPE == 'regular':
            cls.__POOL.append(Tetromino([[1, 0], [2, 0], [0, 1], [1, 1]], 4, -1, 1.0, 0.0, 'S', 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0], [1, 1], [2, 1]], 4, -1, 1.0, 1.0, 'Z', 0, 2))
            cls.__POOL.append(Tetromino([[0, 1], [1, 1], [2, 1], [3, 1]], 3, -1, 1.5, 1.5, 'I', 0, 2))
            cls.__POOL.append(Tetromino([[1, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 1.0, 1.0, 'T', 0, 4))
            cls.__POOL.append(Tetromino([[0, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 0.5, 0.5, 'J', 0, 4))
            cls.__POOL.append(Tetromino([[2, 0], [0, 1], [1, 1], [2, 1]], 4, -1, 1.5, 0.5, 'L', 0, 4))
            cls.__POOL.append(Tetromino([[1, 0], [1, 1], [2, 1], [2, 0]], 3, -1, 1.5, 0.5, 'O', 0, 1))
        # mini Tetris
        elif GAME_TYPE == 'mini':
            cls.__POOL.append(Tetromino([[1, 0], [1, 1]], 0, -1, 1.0, 0.0, 'i', 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [1, 1]], 0, -1, 0.5, 0.5, '/', 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0], [1, 1]], 0, -1, 0.5, 0.5, 'l', 0, 4))

        # extra Tetris
        elif GAME_TYPE == 'extra':
            # Tetromino([[x1,y1],[x2,y2]...], begin_x, begin_y, rotate_center_x, rotate_center_y, name, 0, rotate_max)
            # All [x,y] must be in the range [0,0] (left top) to [4,3] (right, bottom).
            # rotate_max is the possible states of the piece by rotation. e.g., an 'O' piece has only one state, an 'S' piece has two, and an 'L' has four.
            cls.__POOL.append(Tetromino([[1, 1]], 4, -1, 1.0, 1.0, '._extra', 0, 1))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0]], 4, -1, 0.0, 0.0, 'i.extra', 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0], [2, 0]], 4, -1, 1.0, 0.0, '1.extra', 0, 2))
            cls.__POOL.append(Tetromino([[0, 0], [0, 1], [1, 1], [2, 1], [2, 0]], 4, -1, 1.0, 1.0, 'C.extra', 0, 4))
            cls.__POOL.append(Tetromino([[0, 0], [0, 1], [1, 1], [2, 1], [3, 1]], 4, -1, 1.0, 1.0, 'J.extra', 0, 4))
            cls.__POOL.append(Tetromino([[3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], 4, -1, 2.0, 1.0, 'L.extra', 0, 4))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]], 4, -1, 0.5, 0.5, 'Z.extra', 0, 4))
            cls.__POOL.append(Tetromino([[0, 1], [1, 1], [1, 0], [2, 0], [3, 0]], 4, -1, 1.5, 0.5, 'S.extra', 0, 4))
            cls.__POOL.append(Tetromino([[0, 0], [1, 0], [2, 0], [1, 1], [1, 2]], 3, -1, 1.0, 1.0, 'T.extra', 0, 4))
            cls.__POOL.append(
                Tetromino([[1, 0], [1, 1], [1, 2], [2, 2], [3, 2], [3, 1], [3, 0], [2, 0]],
                          3, -1, 2.0, 1.0, 'O.massive', 0, 1))
            cls.__POOL.append(
                Tetromino([[0, 0], [0, 1], [0, 2], [1, 1], [2, 1], [2, 0], [2, 2]], 3, -1, 1, 1, 'H.massive', 0, 2))
            cls.__POOL.append(
                Tetromino([[1, 0], [1, 1], [1, 2], [2, 2], [2, 1], [3, 2], [3, 1], [3, 0], [2, 0]],
                          3, -1, 2.0, 1.0, 'Donut.massive', 0, 1)
            )
            cls.__POOL.append(
                Tetromino([[1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1]],
                          3, -1, 1.0, 1.0, 'Sword.massive', 0, 4)
            )
            # cls.__POOL.append(
            #     Tetromino([[0, 0], [1, 1], [2, 0], [2, 2], [0, 2]],
            #               3, -1, 1.0, 1.0, 'Cross.massive', 0, 1)
            # )

        for tet in cls.__POOL:
            if 'massive' in tet.type_str:
                cls.RNG_THRESHOLD.append(cls.MASSIVE_WEIGHT)
            else:
                cls.RNG_THRESHOLD.append(1.0)

        rng_sum = sum(cls.RNG_THRESHOLD)
        for i in range(len(cls.RNG_THRESHOLD)):
            cls.RNG_THRESHOLD[i] /= rng_sum

        for i in range(1, len(cls.RNG_THRESHOLD)):
            cls.RNG_THRESHOLD[i] += cls.RNG_THRESHOLD[i - 1]

    @classmethod
    def pool_size(cls):
        return len(cls.__POOL)

    @classmethod
    def type_str_to_num(cls, type_str_arg):
        count = 1  # count start from 1 because 0 is reserved for empty
        for tetromino in cls.__POOL:
            if type_str_arg == tetromino.type_str:
                return count
            count += 1

        print("type_str:" + type_str_arg + " not found")
        return None

    @classmethod
    def num_to_type_str(cls, num):
        # num start from 1, because 0 is reserved for empty
        return cls.__POOL[num - 1].type_str

    def __init__(self, tet, start_x, start_y, rot_x, rot_y, type_str_arg, rot, rot_max):
        self.tet = []
        for sq in tet:
            self.tet.append(list(sq))  # make sure this is copy
        self.center_x = start_x
        self.center_y = start_y
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.type_str = type_str_arg
        self.rot = rot
        self.rot_max = rot_max

    @classmethod
    def new_tetromino(cls, type_str_arg):
        for tet in cls.__POOL:
            if tet.type_str == type_str_arg:
                return tet.copy()
        print("type_str is not found")
        return None

    @classmethod
    def new_tetromino_num(cls, type_num):
        return Tetromino.__POOL[type_num].copy()

    @classmethod
    def new_tetromino_fl(cls, rng_fl=None):
        if rng_fl is None:
            rng_fl = random.random()

        for i in range(len(cls.RNG_THRESHOLD)):
            if rng_fl < cls.RNG_THRESHOLD[i]:
                return cls.__POOL[i].copy()

        print('ERROR: rng_fl must be between 0 and 1')
        return None

    @classmethod
    def random_type_str(cls, rng_fl=None):
        return cls.new_tetromino_fl(rng_fl).type_str

    def copy(self):
        return Tetromino(self.tet, self.center_x, self.center_y, self.rot_x, self.rot_y, self.type_str, self.rot,
                         self.rot_max)

    # turn +1 rotate counterclockwise
    def move(self, mov):
        (right, down, turn) = mov
        if (self.type_str == 'S' or self.type_str == 'Z') and turn != 0:
            # for S and Z pieces, it will rotate back if they have been rotated
            if self.rot == 1:
                turn = -1
            else:
                turn = 1

        if turn != 0:
            for sq in self.tet:
                a = sq[0]
                b = sq[1]
                x = self.rot_x
                y = self.rot_y

                sq[0] = round(turn * (b - y) + x)
                sq[1] = round(-turn * (a - x) + y)

        self.center_x += right
        self.center_y += down
        self.rot += turn
        self.rot = self.rot % self.rot_max

        return self

    def to_str(self):
        s = ""
        displaced = self.get_displaced()
        for sq in displaced:
            s += "[" + str(sq[0]) + ", " + str(sq[1]) + "] "
        s += "centerXY: " + str(self.center_x) + ", " + str(self.center_y) + " "
        s += "type: " + self.type_str
        return s

    def to_num(self):
        return self.type_str_to_num(self.type_str)

    def get_displaced(self):
        disp = list()
        for sq in self.tet:
            new_sq = list(sq)
            new_sq[0] = sq[0] + self.center_x
            new_sq[1] = sq[1] + self.center_y
            disp.append(new_sq)
        return disp

    def to_main_grid(self):
        disp = self.get_displaced()
        width = GAME_BOARD_WIDTH
        height = GAME_BOARD_HEIGHT
        row = list()
        grid = list()
        for i in range(width):
            row += [0]
        for j in range(height):
            grid += [list(row)]
        for sq in disp:
            grid[sq[1]][sq[0]] = self.to_num()

        return grid

    def to_above_grid(self):
        disp = self.get_displaced()
        width = GAME_BOARD_WIDTH
        above_grid = [0] * width
        for sq in disp:
            if sq[1] == -1:
                above_grid[sq[0]] = self.to_num()

        return above_grid

    def check_above_grid(self):
        disp = self.get_displaced()
        for sq in disp:
            if sq[1] < 0:
                return True

        return False

    @classmethod
    def to_small_window(cls, type_str):
        small = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        if type_str is None: return small  # if hold is None
        tetro = cls.new_tetromino(type_str)
        for sq in tetro.tet:
            a = sq[0]
            b = sq[1]
            small[b][a] = cls.type_str_to_num(type_str)

        return small


Tetromino.create_pool()
