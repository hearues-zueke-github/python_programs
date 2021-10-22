#! /usr/bin/python3.9

import hashlib
import pdb

import pygame as pg

from enum import Enum

pg.init()

width = 25
height = 25

screen = pg.display.set_mode([width*10, height*10])
clock = pg.time.Clock()

screen_width = screen.get_width()
screen_height = screen.get_height()

print('screen_width: {}'.format(screen_width))
print('screen_height: {}'.format(screen_height))

# left, top, width, height

class Move(Enum):
    NO_MOVE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Player(pg.Rect):
    __slot__ = ['x', 'y', 'w', 'h', 'color']
    def __init__(self, x, y, w, h):
        self.color = (0, 0, 255)
        super(Player, self).__init__(x, y, w, h)

    def draw(self):
        pg.draw.rect(screen, self.color, (self.x, self.y, self.w, self.h), 0)

    def move(self, dx_left: int, dx_right: int, dy_up: int, dy_down: int):
        player.x += (dx_left + dx_right) * width
        player.y += (dy_up + dy_down) * height

class Enemy(pg.Rect):
    __slot__ = ['x', 'y', 'w', 'h', 'color']
    def __init__(self, x: int, y: int, w: int, h: int, player: Player):
        self.digest = b''
        self.seed = 0
        self.last_move = Move.NO_MOVE
        self.player = player
        self.color = (255, 0, 0)
        super(Enemy, self).__init__(x, y, w, h)

    def draw(self):
        pg.draw.rect(screen, self.color, (self.x+1, self.y+1, self.w-2, self.h-2), 0)

    def move_up(self) -> bool:
        if self.y > 0:
            self.y -= height
            return True
        return False

    def move_down(self) -> bool:
        if self.y + height < screen_height:
            self.y += height
            return True
        return False

    def move_left(self) -> bool:
        if self.x > 0:
            self.x -= width
            return True
        return False

    def move_right(self) -> bool:
        if self.x + width < screen_width:
            self.x += width
            return True
        return False

    def calc_next_seed(self):
        byte_arr = self.digest + bytearray(str(player).encode('utf-8'))
        self.digest = hashlib.sha256(byte_arr).digest()
        self.seed = int.from_bytes(self.digest, 'little')

    def move(self):
        is_moved = False
        while not is_moved:
            self.calc_next_seed()
            move_num = self.seed % 4

            print('move_num: {}'.format(move_num))

            if move_num == 0:
                is_moved = self.move_up()
            elif move_num == 1:
                is_moved = self.move_right()
            elif move_num == 2:
                is_moved = self.move_down()
            elif move_num == 3:
                is_moved = self.move_left()

player = Player(x=0*width, y=0*height, w=width, h=height)
enemy = Enemy(x=9*width, y=9*height, w=width, h=height, player=Player)

dx_left = 0
dx_right = 0
dy_up = 0
dy_down = 0

is_key_down = False

# Run until the user asks to quit
running = True
while running:
    clock.tick(10)

    # Did the user click the window close button?
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            is_key_down = True
            if event.key == pg.K_LEFT:
                if player.x > 0:
                    dx_left = -1
            elif event.key == pg.K_RIGHT:
                if player.x + width < screen_width:
                    dx_right = 1
            elif event.key == pg.K_UP:
                if player.y > 0:
                    dy_up = -1
            elif event.key == pg.K_DOWN:
                if player.y + height < screen_height:
                    dy_down = 1
            elif event.key == pg.K_SPACE:
                pdb.set_trace()

            player.move(dx_left=dx_left, dx_right=dx_right, dy_up=dy_up, dy_down=dy_down)
            enemy.move()

            dx_left = 0
            dx_right = 0
            dy_up = 0
            dy_down = 0

            print('player: {}'.format(player))
        elif event.type == pg.KEYUP:
            is_key_down = False

    # if is_key_down:
    #     enemy.move()

    # Fill the background with white
    screen.fill((0, 0, 0))

    # Draw a solid blue circle in the center
    
    player.draw()
    enemy.draw()

    pg.draw.line(screen, (0, 255, 0), (0, player.y+height/2), (screen_width, player.y+height/2), 1)
    pg.draw.line(screen, (0, 255, 0), (player.x+width/2, 0), (player.x+width/2, screen_height), 1)
    # pg.draw.rect(screen, (0, 0, 255), (player.x+10, player.y-20, player.w, player.h), 0)

    # Flip the display
    pg.display.flip()

# Done! Time to quit.
pg.quit()
