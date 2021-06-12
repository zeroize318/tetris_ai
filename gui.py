import pygame
from lib import color
from common import *

class RectC:
    def __init__(self, rect, c):
        self.rect = rect
        self.color = c


MAIN_GRID_BEGIN_CORNER = (330, 19)
ABOVE_GRID_BEGIN_CORNER = (330, -15)
HOLD_GRID_BEGIN_CORNER = (210, 19)
NEXT_GRID_BEGIN_CORNER = (669, 19)
NEXT_GRID_Y_DIFF = 80

SCORE_BOARD_RECTC = RectC((669, 459, 200, 200), color.LIGHT_GRAY)
INFO_BOARD_RECTC = RectC((110, 459, 200, 200), color.LIGHT_GRAY)

WIN_SIZE = (1000, 700)

pygame.init()
FONT_SIZE = 16
FONT = pygame.font.SysFont('couriernew', FONT_SIZE)            # 'couriernewbold', 'osaka'

class Gui:
    def __init__(self, is_display=True, delay=50):
        self.is_display = is_display
        self.delay = delay

        if not self.is_display: return

        # setup window
        self.win = pygame.display.set_mode(WIN_SIZE)
        pygame.display.set_caption('tetris_ai')

        # setup grids
        self.main_grid_rectc = Gui.__create_grid_rectc__(MAIN_GRID_BEGIN_CORNER, 32, 2, GAME_BOARD_HEIGHT,
                                                         GAME_BOARD_WIDTH)
        self.above_grid_rectc = Gui.__create_grid_rectc__(ABOVE_GRID_BEGIN_CORNER, 32, 2, 1, GAME_BOARD_WIDTH)
        self.hold_grid_rectc = Gui.__create_grid_rectc__(HOLD_GRID_BEGIN_CORNER, 25, 1, 3, 4)
        self.next_grid_rectc = list()
        for i in range(5):
            begin = list(NEXT_GRID_BEGIN_CORNER)
            begin[1] = begin[1] + i * NEXT_GRID_Y_DIFF
            begin = tuple(begin)
            self.next_grid_rectc += [Gui.__create_grid_rectc__(begin, 25, 1, 3, 4)]

        # setup text panels
        self.__score_board_text__ = "score board \ntesting"
        self.__info_board_text__ = "info board \ntesting"


    @staticmethod
    def __create_grid_rectc__(begin_corner, size, gap, height, width):
        grid = []
        for i in range(height):
            row = []
            for j in range(width):
                x = begin_corner[0] + j * size + gap
                y = begin_corner[1] + i * size + gap
                row += [RectC(pygame.Rect(x, y, size - gap * 2, size - gap * 2), color.cmap[0])]
            grid += [row]

        return grid

    def redraw(self):
        if not self.is_display: return

        try:
            pygame.time.delay(self.delay)
        except KeyboardInterrupt:
            pass

        self.win.fill(color.DARK_GRAY)
        self.__paint_grids__()
        self.__paint_panels__()

        pygame.display.update()

    def __paint_grids__(self):
        for row in self.main_grid_rectc:
            for rc in row:
                pygame.draw.rect(self.win, rc.color, rc.rect)

        for row in self.hold_grid_rectc:
            for rc in row:
                pygame.draw.rect(self.win, rc.color, rc.rect)

        for grid in self.next_grid_rectc:
            for row in grid:
                for rc in row:
                    pygame.draw.rect(self.win, rc.color, rc.rect)

        for row in self.above_grid_rectc:
            for rc in row:
                pygame.draw.rect(self.win, rc.color, rc.rect)

    def update_grids_color(self, grids_int, above_grid=None):
        main_int, hold_int, next_int = grids_int
        r = 0
        for row in self.main_grid_rectc:
            c = 0
            for rc in row:
                rc.color = color.cmap[main_int[r][c]]
                c += 1
            r += 1

        r = 0
        for row in self.hold_grid_rectc:
            c = 0
            for rc in row:
                rc.color = color.cmap[hold_int[r][c]]
                c += 1
            r += 1

        g = 0
        for grid in self.next_grid_rectc:
            r = 0
            for row in grid:
                c = 0
                for rc in row:
                    rc.color = color.cmap[next_int[g][r][c]]
                    c += 1
                r += 1
            g += 1

        if above_grid is not None:
            c = 0
            for rc in self.above_grid_rectc[0]:
                if above_grid[c] == 0:
                    rc.color = color.DARK_GRAY
                else:
                    rc.color = color.cmap[above_grid[c]]

                c += 1

    def __paint_panels__(self):
        # score text
        pygame.draw.rect(self.win, SCORE_BOARD_RECTC.color, SCORE_BOARD_RECTC.rect)
        lines = self.__score_board_text__.split('\n')
        i = 0
        for line in lines:
            text = FONT.render(line, True, color.BLACK)
            self.win.blit(text, (671, 461 + i * FONT_SIZE))
            i += 1

        # info text
        pygame.draw.rect(self.win, INFO_BOARD_RECTC.color, INFO_BOARD_RECTC.rect)
        lines = self.__info_board_text__.split('\n')
        i = 0
        for line in lines:
            text = FONT.render(line, True, color.BLACK)
            self.win.blit(text, (112, 461 + i * FONT_SIZE))
            i += 1

    def set_score_text(self, score_text):
        self.__score_board_text__ = score_text

    def set_info_text(self, info_text):
        self.__info_board_text__ = info_text


if __name__ == "__main__":
    gui = Gui()

    test_main_grid = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                      [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                      [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                      [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                      [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                      [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                      [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                      [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                      [9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                      [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                      [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                      [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                      [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                      [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                      [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                      [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                      [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]

    test_hold_grid = [[0, 1, 2, 3],
                      [1, 2, 3, 0],
                      [2, 3, 0, 1]]

    test_next_grid = [[[0, 1, 2, 3],
                       [1, 2, 3, 0],
                       [2, 3, 0, 1]],
                      [[0, 1, 2, 3],
                       [1, 2, 3, 0],
                       [2, 3, 0, 1]],
                      [[0, 1, 2, 3],
                       [1, 2, 3, 0],
                       [2, 3, 0, 1]],
                      [[0, 1, 2, 3],
                       [1, 2, 3, 0],
                       [2, 3, 0, 1]],
                      [[0, 1, 2, 3],
                       [1, 2, 3, 0],
                       [2, 3, 0, 1]]]

    gui.update_grids_color((test_main_grid, test_hold_grid, test_next_grid))

    is_run = True

    while is_run:
        gui.redraw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_run = False
