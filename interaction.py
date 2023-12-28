import pygame
import random
import csv

RES = 16
DIMS = (28, 28)
MAXCOLOR = 255
SCREEN = (RES * DIMS[0], RES * DIMS[1])
display = pygame.display.set_mode(SCREEN)

class Tile:

    def __init__(self, c, r) -> None:

        self.c = c
        self.r = r
        self.x = self.c * RES
        self.y = self.r * RES
        self.state = 0
        self.rect = pygame.Rect(self.x, self.y, RES, RES)

    def getColor(self):
        return pygame.Color(self.state, self.state, self.state)

    def draw(self):
        pygame.draw.rect(display, self.getColor(), self.rect)

grid = []
for r in range(DIMS[1]):
    row = []
    for c in range(DIMS[0]):
        tile = Tile(c, r)
        row.append(tile)
    grid.append(row)

def draw():
    display.fill("red")
    for row in grid:
        for tile in row:
            tile.draw()
    pygame.display.flip()

def saveImageIntoCSV(label, gridArr: list[list[Tile]]) -> None:
    length = len(gridArr)
    with open('data/test.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        newRow = [gridArr[i][j].state for j in range(length) for i in range(length)]
        newRow.insert(0, label)
        writer.writerow(newRow)

def reset() -> None:
    for row in grid:
        for tile in row:
            tile.state = 0

isMouseDown = False
def update():
    global isMouseDown
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            isMouseDown = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            isMouseDown = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                saveImageIntoCSV(0, grid)
                reset()
            elif event.key == pygame.K_1:
                saveImageIntoCSV(1, grid)
                reset()
            elif event.key == pygame.K_2:
                saveImageIntoCSV(2, grid)
                reset()
            elif event.key == pygame.K_3:
                saveImageIntoCSV(3, grid)
                reset()
            elif event.key == pygame.K_4:
                saveImageIntoCSV(4, grid)
                reset()
            elif event.key == pygame.K_5:
                saveImageIntoCSV(5, grid)
                reset()
            elif event.key == pygame.K_6:
                saveImageIntoCSV(6, grid)
                reset()
            elif event.key == pygame.K_7:
                saveImageIntoCSV(7, grid)
                reset()
            elif event.key == pygame.K_8:
                saveImageIntoCSV(8, grid)
                reset()
            elif event.key == pygame.K_9:
                saveImageIntoCSV(9, grid)
                reset()

    if isMouseDown == True:
        x, y = pygame.mouse.get_pos()
        for i, row in enumerate(grid):
            for j, tile in enumerate(row):
                if tile.rect.collidepoint((x,y)):
                    tile.state = MAXCOLOR

                    # Drawing small highlights around cursor
                    if i > 0:
                        grid[i-1][j].state = max(75, grid[i-1][j].state)
                    if i < DIMS[0]-1:
                        grid[i+1][j].state = max(75, grid[i+1][j].state)
                    if j > 0:
                        grid[i][j-1].state = max(75, grid[i][j-1].state)
                    if j < DIMS[1]-1:
                        grid[i][j+1].state = max(75, grid[i][j+1].state)


if __name__ == "__main__":
    while True:
        draw()
        update()