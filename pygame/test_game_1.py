# Simple pg program

# Import and initialize the pg library
import pygame as pg
pg.init()

# Set up the drawing window
screen = pg.display.set_mode([500, 500])
clock = pg.time.Clock()

class Circle(Exception):
    def __init__(self, x, y):
        self.x = x
        self.y = y

circ1 = Circle(250, 140)
dx_left = 0
dx_right = 0
dy_up = 0
dy_down = 0

vel = 1

# Run until the user asks to quit
running = True
while running:
    clock.tick(60)


    # Did the user click the window close button?
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                dx_left = -1
            elif event.key == pg.K_RIGHT:
                dx_right = 1
            elif event.key == pg.K_UP:
                dy_up = -1
            elif event.key == pg.K_DOWN:
                dy_down = 1
            elif event.key == pg.K_LSHIFT or event.key == pg.K_RSHIFT:
                vel = 3
        elif event.type == pg.KEYUP:
            if event.key == pg.K_LEFT:
                dx_left = 0
            elif event.key == pg.K_RIGHT:
                dx_right = 0
            elif event.key == pg.K_UP:
                dy_up = 0
            elif event.key == pg.K_DOWN:
                dy_down = 0
            elif event.key == pg.K_LSHIFT or event.key == pg.K_RSHIFT:
                vel = 1

    # print('dx_left: {}, dx_right_ {}'.format(dx_left, dx_right))
    circ1.x += (dx_left + dx_right) * vel
    circ1.y += (dy_up + dy_down) * vel

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pg.draw.circle(screen, (0, 0, 255), (circ1.x, circ1.y), 75)

    # Flip the display
    pg.display.flip()

# Done! Time to quit.
pg.quit()
