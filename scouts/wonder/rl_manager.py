from collections import deque
import enum

class States(enum.Enum):
    UNKNOWN = -1  # Placeholder for infinite state value
    EMPTY = 0
    RECON = 1
    MISSION = 5
    WALL = float('inf')

class RLManager:
    def __init__(self, map_size = 16 * 2 + 1):
        self.map_size = map_size
        self.global_map = [[States.UNKNOWN for _ in range(map_size)] for _ in range(map_size)]
        self.recent_targets = deque(maxlen=10)

        for i in range(map_size):
            self.global_map[0][i] = States.WALL                   # Top row
            self.global_map[map_size - 1][i] = States.WALL        # Bottom row
            self.global_map[i][0] = States.WALL                   # Left column
            self.global_map[i][map_size - 1] = States.WALL        # Right column
        
        for i in range(0, map_size, 2):
            for j in range(0, map_size, 2):
                self.global_map[i][j] = States.WALL

    def parse_viewcone_tile(self, viewcone_tile: int, dir: int) -> dict:
        def rotate_right(arr, x):
            x = x % len(arr)  # handle x > len(arr)
            return arr[-x:] + arr[:-x]

        last2 = viewcone_tile & 0b11
        state = States.UNKNOWN
        if last2 == 0b00:
            state = States.UNKNOWN
        elif last2 == 0b01:
            state = States.EMPTY
        elif last2 == 0b10:
            state = States.RECON
        elif last2 == 0b11:
            state = States.MISSION
        # these are wall directions relative to agent's direction
        # so we need to offset them by the agent's direction
        #           right                          bottom                   left                        top
        walls = [viewcone_tile & 0b10000, viewcone_tile & 0b100000, viewcone_tile & 0b1000000, viewcone_tile & 0b10000000]
        no_rotations = dir % 4
        walls = rotate_right(walls, no_rotations)
        walls = [bool(i) for i in walls]

        return {
            "state": state,
            "scout": True if (viewcone_tile & 0b100) else False,
            "guard": True if (viewcone_tile & 0b1000) else False,
            "walls": walls,
        }

    def _viewcone_to_map_coords(self, vc_row, vc_col, agent_location, agent_direction):
        """
        Translates viewcone coordinates relative to the agent into global map coordinates.

        Args:
            vc_row (int): Row index in the 7x5 viewcone (0-6).
            vc_col (int): Column index in the 7x5 viewcone (0-4).
            agent_location (list): Agent's global location [x, y].
            agent_direction (int): Agent's direction (0:E, 1:S, 2:W, 3:N).

        Returns:
            tuple: Global map coordinates (map_x, map_y) or None if outside map bounds.
        """
        # Agent is at viewcone[2, 2]
        relative_row = vc_row - 2
        relative_col = vc_col - 2

        if agent_direction == 3: # North -> (0,0) is bottom left relative to agent
            delta_x = relative_col
            delta_y = -relative_row
        elif agent_direction == 0: # East -> (0,0) is top left relative to agent
            delta_x = relative_row
            delta_y = relative_col
        elif agent_direction == 1: # South -> (0,0) is top right relative to agent
            delta_x = -relative_col
            delta_y = relative_row
        elif agent_direction == 2: # West -> (0,0) is bottom right relative to agent
            delta_x = -relative_row
            delta_y = -relative_col
        else:
            raise ValueError(f"Invalid agent direction: {agent_direction}")

        # Calculate global map coordinates
        map_x = agent_location[0] + delta_x
        map_y = agent_location[1] + delta_y

        # Check if coordinates are within map bounds
        if 0 <= map_x < 16 and 0 <= map_y < 16:
            return (map_x, map_y)
        else:
            return None # Outside map bounds
    
    def print_map(self, scout, agents=[]):
        for i in range(self.map_size):
            for j in range(self.map_size):
                if scout == (i, j):
                    print("S", end="")
                    continue
                elif (i, j) in agents:
                    print("A", end="")
                    continue
                cell = self.global_map[i][j]
                if cell == States.EMPTY:
                    print(" ", end="")
                elif cell == States.RECON:
                    print("R", end="")
                elif cell == States.MISSION:
                    print("M", end="")
                elif cell == States.WALL:
                    print("#", end="")
                else:
                    print("?", end="")
            print()
    
    def update_global_map(self, new_tile, x, y):
        walls = new_tile["walls"]
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for idx, (dx, dy) in enumerate(neighbors):
            nx, ny = x + dx, y + dy
            if walls[idx]:
                self.global_map[nx][ny] = States.WALL
            elif self.global_map[nx][ny] == States.UNKNOWN:
                self.global_map[nx][ny] = States.EMPTY

        self.global_map[x][y] = new_tile["state"]

    def _evaluate_target_score(self, x, y):
        state = self.global_map[x][y]
        if state == States.MISSION:
            return 100
        elif state == States.RECON:
            return 50
        elif state == States.UNKNOWN:
            unknown_neighbors = 0
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.global_map[nx][ny] == States.UNKNOWN:
                        unknown_neighbors += 1
            return 10 + unknown_neighbors
        return -1

    def plan_next_action(self, current_pos, current_dir, avoid_positions):
        dir_to_delta = {
            0: (2, 0),
            1: (0, 2),
            2: (-2, 0),
            3: (0, -2),
        }

        visited = set()
        queue = deque([(current_pos[0], current_pos[1], current_dir, [])])
        candidates = []

        def is_valid(x, y):
            return 0 <= x < self.map_size and 0 <= y < self.map_size and self.global_map[x][y] != States.WALL

        def is_transition_clear(x, y, dx, dy):
            mid_x, mid_y = x + dx // 2, y + dy // 2
            dest_x, dest_y = x + dx, y + dy
            return (
                is_valid(dest_x, dest_y) and
                is_valid(mid_x, mid_y) and
                (dest_x, dest_y) not in avoid_positions
            )

        def turn_cost(actions):
            cost = 0
            for a in actions:
                if a == 0:
                    cost += 1
                elif a in [2, 3]:
                    cost += 2
                elif a == 1:
                    cost += 3
            return cost

        while queue:
            x, y, facing, actions = queue.popleft()
            if (x, y, facing) in visited:
                continue
            visited.add((x, y, facing))

            if (x, y) != current_pos and (x, y) not in self.recent_targets:
                score = self._evaluate_target_score(x, y)
                if score > 0:
                    candidates.append((score - turn_cost(actions), actions))

            # Move forward
            dx, dy = dir_to_delta[facing]
            if is_transition_clear(x, y, dx, dy):
                queue.append((x + dx, y + dy, facing, actions + [0]))

            # Turn left
            new_dir = (facing + 3) % 4
            queue.append((x, y, new_dir, actions + [2]))

            # Turn right
            new_dir = (facing + 1) % 4
            queue.append((x, y, new_dir, actions + [3]))

            # Move backward
            back_dir = (facing + 2) % 4
            bdx, bdy = dir_to_delta[back_dir]
            if is_transition_clear(x, y, bdx, bdy):
                queue.append((x + bdx, y + bdy, facing, actions + [1]))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_actions = candidates[0][1]
            if best_actions:
                self.recent_targets.append(current_pos)
                return best_actions[0]

        return 0  # wait fallback

    def plan_guard_action(self, current_pos, current_dir, scout_pos=None):
        dir_to_delta = {
            0: (2, 0),
            1: (0, 2),
            2: (-2, 0),
            3: (0, -2),
        }

        visited = set()
        queue = deque([(current_pos[0], current_pos[1], current_dir, [])])
        candidates = []

        def is_valid(x, y):
            return 0 <= x < self.map_size and 0 <= y < self.map_size and self.global_map[x][y] != States.WALL

        def is_transition_clear(x, y, dx, dy):
            mid_x, mid_y = x + dx // 2, y + dy // 2
            dest_x, dest_y = x + dx, y + dy
            return is_valid(dest_x, dest_y) and is_valid(mid_x, mid_y)

        def is_goal(x, y):
            if scout_pos:
                return (x, y) == scout_pos
            return self.global_map[x][y] in [States.UNKNOWN, States.RECON, States.MISSION]

        def turn_cost(actions):
            cost = 0
            for a in actions:
                if a == 0:
                    cost += 1
                elif a in [2, 3]:
                    cost += 2
                elif a == 1:
                    cost += 3
            return cost

        while queue:
            x, y, facing, actions = queue.popleft()
            if (x, y, facing) in visited:
                continue
            visited.add((x, y, facing))

            if is_goal(x, y) and (x, y) != tuple(current_pos):
                score = self._evaluate_target_score(x, y)
                if score > 0:
                    candidates.append((score - turn_cost(actions), actions))

            # Move forward
            dx, dy = dir_to_delta[facing]
            if is_transition_clear(x, y, dx, dy):
                queue.append((x + dx, y + dy, facing, actions + [0]))

            # Turn left
            new_dir = (facing + 3) % 4
            queue.append((x, y, new_dir, actions + [2]))

            # Turn right
            new_dir = (facing + 1) % 4
            queue.append((x, y, new_dir, actions + [3]))

            # Move backward
            back_dir = (facing + 2) % 4
            bdx, bdy = dir_to_delta[back_dir]
            if is_transition_clear(x, y, bdx, bdy):
                queue.append((x + bdx, y + bdy, facing, actions + [1]))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_actions = candidates[0][1]
            if best_actions:
                self.recent_targets.append(current_pos)
                return best_actions[0]

        return 0  # wait fallback

    def rl(self, observation_json: dict) -> int:
        instance = observation_json
        viewcone = instance["viewcone"]
        direction = instance["direction"]
        location = instance["location"]
        is_scout = instance["scout"]
        step = instance["step"]

        # print(location, direction)

        viewcone_info = [
            [self.parse_viewcone_tile(int(tile), int(direction)) for tile in row]
            for row in viewcone
        ]

        scout = None
        guards = []
        for vc_row in range(7):
            for vc_col in range(5):
                res = self._viewcone_to_map_coords(vc_row, vc_col, location, direction)
                if res:
                    gx, gy = res
                    self.update_global_map(viewcone_info[vc_row][vc_col], gx * 2 + 1, gy * 2 + 1)
                    # print(vc_row, vc_col, viewcone_info[vc_row][vc_col]["scout"])
                    if viewcone_info[vc_row][vc_col]["scout"]:
                        scout = (gx * 2 + 1, gy * 2 + 1)
                    elif viewcone_info[vc_row][vc_col]["guard"]:
                        guards.append((gx * 2 + 1, gy * 2 + 1))

        # self.print_map(scout, guards)

        if is_scout:
            # print(location, scout)
            answer = self.plan_next_action(scout, direction, avoid_positions=guards)
            # print("Answer:", answer)
            return answer
        else:
            return self.plan_guard_action(current_pos=location, current_dir=direction)