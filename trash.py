def dfs(self):
    stacked = []
    visited = []
    do_dfs = True
    stacked.append([self.player.x, self.player.y])
    print("stacked: ", stacked)
    while do_dfs:
        self.next_visit = stacked.pop()
        visited.append([self.player.x, self.player.y])
        print("next: ", self.next_visit)
        # move to next place
        if [self.player.x, self.player.y] != self.next_visit:
            self.player.x = self.next_visit[0]
            self.player.y = self.next_visit[1]

        # stack next places to visit
        nearby_cells = [[self.player.x - 8, self.player.y], [self.player.x, self.player.y - 8],
                        [self.player.x + 8, self.player.y], [self.player.x, self.player.y + 8]]
        for cells in nearby_cells:
            if 8 <= cells[0] <= 48 and 8 <= cells[1] <= 48:
                print("check: ", cells)
                if not self._check_wall_manual(cells):
                    if cells not in stacked and cells not in visited:
                        print("append: ", cells)
                        stacked.append(cells)

        print("stacked: ", stacked)
        print("visited: ", visited)

        if len(stacked) == 0 or len(visited) == 20 or self.next_visit == [40, 24]:
            do_dfs = False
            self._reset()

def _check_wall_manual(self, coords):
    is_wall = False
    walls = [[8, 32], [8, 40], [16, 24], [32, 16], [40, 16]]
    if coords[0] * coords[1] <= 0 or coords[0] >= 48 or coords[1] >= 48:
        is_wall = True
    if coords in walls:
        is_wall = True

    return is_wall

def move_dfs(self):
    if self.move_count == 0:
        self.move_count = 8

def update_dfs(self):
    if pyxel.btn(pyxel.KEY_1):
        self.dfs()

def _check_wall(self, item):
    return pyxel.tilemap(0).get(math.floor(item.x / 7) + self.move_x, math.floor(item.y / 7) + self.move_y) >= 64

def _check_wall2(self, item, check_dirc):
    x = math.floor(item.x / 7) + check_dirc[0] * 8
    y = math.floor(item.y / 7) + check_dirc[1] * 8
    if x * y <= 0:
        return False
    return pyxel.tilemap(0).get(x, y) >= 64
