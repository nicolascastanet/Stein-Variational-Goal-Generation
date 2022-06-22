import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, glob
import imageio


def extract_success_data(base_path, timestep_dict, timestep_list, min_timestep, max_timestep, folder_name, episode_length, scale, skip_every, filter_unique=False, size=None):
    goals_list = []

    # Skip every other and up to 120k 
    for timestep in timestep_list[::skip_every]:
        #print(timestep)
        if timestep < max_timestep and timestep > min_timestep:
            path = os.path.join(base_path, timestep_dict[timestep], folder_name)
            #print(path)
            if os.path.exists(path):
                filename = os.path.join(path, "tensors.tsv")
                data = np.genfromtxt(fname=filename, delimiter="\t", skip_header=0, filling_values=-1)  # change filling_values as req'd to fill in missing values
                # Filter out repeated datapoints
                if size is not None:
                    goals_list.append(data[:size])
                else:
                    goals_list.append(data)
            else: 
                print("pb ",timestep,path)

    goals_list = np.stack(goals_list, axis=0)
    return goals_list

class Maze:
    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None):
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(self._segments['origin']['loc'], direction))
        self._last_segment = 'origin'
        self.goal_squares = None

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, (tuple, list)):
            self._goal_squares = [gs.lower() for gs in goal_squares]
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(start_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(start_squares, (tuple, list)):
            self.start_squares = [ss.lower() for ss in start_squares]
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(name=this_name.lower(), anchor=last_name, direction=direction)
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments

        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        for segment in self._segments.values():
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
        for x, y in self._walls:
            ax.plot(x, y, 'k-')

    def sample_start(self):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def sample_goal(self, min_wall_dist=None):
        if min_wall_dist is None:
            min_wall_dist = 0.1
        else:
            min_wall_dist = min(0.4, max(0.01, min_wall_dist))

        g_square = self.goal_squares[np.random.randint(low=0, high=len(self.goal_squares))]
        g_square_loc = self._segments[g_square]['loc']
        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = g_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        #assert (float(loc_x0), float(loc_y0)) in self._locs
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5)) for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5)) for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(np.random.rand() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth+1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        return cx + dx, cy + dy

def make_snail_maze(length):
    length = int(length)
    assert length >= 1

    segments = []
    last = 'origin'
    l = length
    t = 1
    i=0
    while l > 1:
        if last == 'origin':
            t_init = 1
        else:
            t_init = t+i-1
            
        for x in range(t_init, l+1):
            next_name = '{},{}'.format(length-l,x)
            segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
            last = str(next_name)
                
        for x in range(t+i, l+1):
            next_name = '{},{}'.format(x,l)
            segments.append({'anchor': last, 'direction': 'up', 'name': next_name})
            last = str(next_name)
                 
        for x in range(t+i, l+1):
            next_name = '{},{}'.format(l+1,x)
            segments.append({'anchor': last, 'direction': 'left', 'name': next_name})
            last = str(next_name)
            
        for x in range(t+i, l):
            next_name = '{},{}'.format(x,length-l)
            segments.append({'anchor': last, 'direction': 'down', 'name': next_name})
            last = str(next_name)
            
        l-=1
        i+=1
        
    return Maze(*segments, goal_squares=last)

mazes_dict = dict()
segments_crazy = [
    {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
     {'anchor': 'origin', 'direction': 'up', 'name': '0,1'},
     {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
     {'anchor': '0,1', 'direction': 'up', 'name': '0,2'},
     {'anchor': '0,2', 'direction': 'right', 'name': '1,2'},
     {'anchor': '2,0', 'direction': 'up', 'name': '2,1'},
     {'anchor': '1,2', 'direction': 'right', 'name': '2,2'},
     {'anchor': '0,2', 'direction': 'up', 'name': '0,3'},
     {'anchor': '2,1', 'direction': 'right', 'name': '3,1'},
     {'anchor': '1,2', 'direction': 'down', 'name': '1,1'},
     {'anchor': '3,1', 'direction': 'down', 'name': '3,0'},
     {'anchor': '1,2', 'direction': 'up', 'name': '1,3'},
     {'anchor': '3,1', 'direction': 'right', 'name': '4,1'},
     {'anchor': '1,3', 'direction': 'up', 'name': '1,4'},
     {'anchor': '4,1', 'direction': 'right', 'name': '5,1'},
     {'anchor': '4,1', 'direction': 'up', 'name': '4,2'},
     {'anchor': '5,1', 'direction': 'down', 'name': '5,0'},
     {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
     {'anchor': '1,4', 'direction': 'right', 'name': '2,4'},
     {'anchor': '4,2', 'direction': 'right', 'name': '5,2'},
     {'anchor': '2,4', 'direction': 'right', 'name': '3,4'},
     {'anchor': '3,4', 'direction': 'up', 'name': '3,5'},
     {'anchor': '1,4', 'direction': 'left', 'name': '0,4'},
     {'anchor': '1,4', 'direction': 'up', 'name': '1,5'},
     {'anchor': '2,2', 'direction': 'up', 'name': '2,3'},
     {'anchor': '3,1', 'direction': 'up', 'name': '3,2'},
     {'anchor': '5,0', 'direction': 'right', 'name': '6,0'},
     {'anchor': '3,2', 'direction': 'up', 'name': '3,3'},
     {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
     {'anchor': '6,0', 'direction': 'up', 'name': '6,1'},
     {'anchor': '6,0', 'direction': 'right', 'name': '7,0'},
     {'anchor': '6,1', 'direction': 'right', 'name': '7,1'},
     {'anchor': '3,4', 'direction': 'right', 'name': '4,4'},
     {'anchor': '1,5', 'direction': 'right', 'name': '2,5'},
     {'anchor': '7,1', 'direction': 'up', 'name': '7,2'},
     {'anchor': '1,5', 'direction': 'up', 'name': '1,6'},
     {'anchor': '4,4', 'direction': 'right', 'name': '5,4'},
     {'anchor': '5,4', 'direction': 'down', 'name': '5,3'},
     {'anchor': '0,4', 'direction': 'up', 'name': '0,5'},
     {'anchor': '7,2', 'direction': 'left', 'name': '6,2'},
     {'anchor': '1,6', 'direction': 'left', 'name': '0,6'},
     {'anchor': '7,0', 'direction': 'right', 'name': '8,0'},
     {'anchor': '7,2', 'direction': 'right', 'name': '8,2'},
     {'anchor': '2,5', 'direction': 'up', 'name': '2,6'},
     {'anchor': '8,0', 'direction': 'up', 'name': '8,1'},
     {'anchor': '3,5', 'direction': 'up', 'name': '3,6'},
     {'anchor': '6,2', 'direction': 'up', 'name': '6,3'},
     {'anchor': '6,3', 'direction': 'right', 'name': '7,3'},
     {'anchor': '3,5', 'direction': 'right', 'name': '4,5'},
     {'anchor': '7,3', 'direction': 'up', 'name': '7,4'},
     {'anchor': '6,3', 'direction': 'up', 'name': '6,4'},
     {'anchor': '6,4', 'direction': 'up', 'name': '6,5'},
     {'anchor': '8,1', 'direction': 'right', 'name': '9,1'},
     {'anchor': '8,2', 'direction': 'right', 'name': '9,2'},
     {'anchor': '2,6', 'direction': 'up', 'name': '2,7'},
     {'anchor': '8,2', 'direction': 'up', 'name': '8,3'},
     {'anchor': '6,5', 'direction': 'left', 'name': '5,5'},
     {'anchor': '5,5', 'direction': 'up', 'name': '5,6'},
     {'anchor': '7,4', 'direction': 'right', 'name': '8,4'},
     {'anchor': '8,4', 'direction': 'right', 'name': '9,4'},
     {'anchor': '0,6', 'direction': 'up', 'name': '0,7'},
     {'anchor': '2,7', 'direction': 'up', 'name': '2,8'},
     {'anchor': '7,4', 'direction': 'up', 'name': '7,5'},
     {'anchor': '9,4', 'direction': 'down', 'name': '9,3'},
     {'anchor': '9,4', 'direction': 'up', 'name': '9,5'},
     {'anchor': '2,7', 'direction': 'left', 'name': '1,7'},
     {'anchor': '4,5', 'direction': 'up', 'name': '4,6'},
     {'anchor': '9,1', 'direction': 'down', 'name': '9,0'},
     {'anchor': '6,5', 'direction': 'up', 'name': '6,6'},
     {'anchor': '3,6', 'direction': 'up', 'name': '3,7'},
     {'anchor': '1,7', 'direction': 'up', 'name': '1,8'},
     {'anchor': '3,7', 'direction': 'right', 'name': '4,7'},
     {'anchor': '2,8', 'direction': 'up', 'name': '2,9'},
     {'anchor': '2,9', 'direction': 'left', 'name': '1,9'},
     {'anchor': '7,5', 'direction': 'up', 'name': '7,6'},
     {'anchor': '1,8', 'direction': 'left', 'name': '0,8'},
     {'anchor': '6,6', 'direction': 'up', 'name': '6,7'},
     {'anchor': '0,8', 'direction': 'up', 'name': '0,9'},
     {'anchor': '7,5', 'direction': 'right', 'name': '8,5'},
     {'anchor': '6,7', 'direction': 'left', 'name': '5,7'},
     {'anchor': '2,9', 'direction': 'right', 'name': '3,9'},
     {'anchor': '3,9', 'direction': 'right', 'name': '4,9'},
     {'anchor': '7,6', 'direction': 'right', 'name': '8,6'},
     {'anchor': '3,7', 'direction': 'up', 'name': '3,8'},
     {'anchor': '9,5', 'direction': 'up', 'name': '9,6'},
     {'anchor': '7,6', 'direction': 'up', 'name': '7,7'},
     {'anchor': '5,7', 'direction': 'up', 'name': '5,8'},
     {'anchor': '3,8', 'direction': 'right', 'name': '4,8'},
     {'anchor': '8,6', 'direction': 'up', 'name': '8,7'},
     {'anchor': '5,8', 'direction': 'right', 'name': '6,8'},
     {'anchor': '7,7', 'direction': 'up', 'name': '7,8'},
     {'anchor': '4,9', 'direction': 'right', 'name': '5,9'},
     {'anchor': '8,7', 'direction': 'right', 'name': '9,7'},
     {'anchor': '7,8', 'direction': 'right', 'name': '8,8'},
     {'anchor': '8,8', 'direction': 'up', 'name': '8,9'},
     {'anchor': '5,9', 'direction': 'right', 'name': '6,9'},
     {'anchor': '6,9', 'direction': 'right', 'name': '7,9'},
     {'anchor': '8,9', 'direction': 'right', 'name': '9,9'},
     {'anchor': '9,9', 'direction': 'down', 'name': '9,8'}
]
mazes_dict['square_large'] = {'maze': Maze(*segments_crazy, goal_squares='9,9'), 'action_range': 0.95}

segments_med_0 = [
     {'name': '1,0', 'anchor': 'origin', 'direction': 'right'},
     {'name': '1,1', 'anchor': '1,0', 'direction': 'up'},
     {'name': '0,1', 'anchor': 'origin', 'direction': 'up'},
     {'name': '2,0', 'anchor': '1,0', 'direction': 'right'},
     {'name': '0,2', 'anchor': '0,1', 'direction': 'up'},
     {'name': '3,0', 'anchor': '2,0', 'direction': 'right'},
     {'name': '2,1', 'anchor': '2,0', 'direction': 'up'},
     {'name': '2,2', 'anchor': '2,1', 'direction': 'up'},
     {'name': '4,0', 'anchor': '3,0', 'direction': 'right'},
     {'name': '3,1', 'anchor': '2,1', 'direction': 'right'},
     {'name': '3,2', 'anchor': '2,2', 'direction': 'right'},
     {'name': '4,1', 'anchor': '3,1', 'direction': 'right'},
     {'name': '1,2', 'anchor': '2,2', 'direction': 'left'},
     {'name': '0,3', 'anchor': '0,2', 'direction': 'up'},
     {'name': '5,0', 'anchor': '4,0', 'direction': 'right'},
     {'name': '0,4', 'anchor': '0,3', 'direction': 'up'},
     {'name': '4,2', 'anchor': '3,2', 'direction': 'right'},
     {'name': '4,3', 'anchor': '4,2', 'direction': 'up'},
     {'name': '3,3', 'anchor': '3,2', 'direction': 'up'},
     {'name': '1,3', 'anchor': '1,2', 'direction': 'up'},
     {'name': '1,4', 'anchor': '1,3', 'direction': 'up'},
     {'name': '2,3', 'anchor': '3,3', 'direction': 'left'},
     {'name': '5,1', 'anchor': '4,1', 'direction': 'right'},
     {'name': '0,5', 'anchor': '0,4', 'direction': 'up'},
     {'name': '2,4', 'anchor': '2,3', 'direction': 'up'},
     {'name': '5,3', 'anchor': '4,3', 'direction': 'right'},
     {'name': '4,4', 'anchor': '4,3', 'direction': 'up'},
     {'name': '5,2', 'anchor': '5,1', 'direction': 'up'},
     {'name': '4,5', 'anchor': '4,4', 'direction': 'up'},
     {'name': '3,4', 'anchor': '4,4', 'direction': 'left'},
     {'name': '2,5', 'anchor': '2,4', 'direction': 'up'},
     {'name': '5,5', 'anchor': '4,5', 'direction': 'right'},
     {'name': '1,5', 'anchor': '2,5', 'direction': 'left'},
     {'name': '3,5', 'anchor': '3,4', 'direction': 'up'},
     {'name': '5,4', 'anchor': '5,3', 'direction': 'up'}]
mazes_dict['square_med_0'] = {'maze': Maze(*segments_med_0, goal_squares='5,5'), 'action_range': 0.95}

segments_pbcs_0 = [
    {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
    {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
    {'anchor': '2,0', 'direction': 'right', 'name': '3,0'},
    {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
    {'anchor': '4,0', 'direction': 'up', 'name': '4,1'},
    {'anchor': '4,1', 'direction': 'left', 'name': '3,1'},
    {'anchor': '3,1', 'direction': 'left', 'name': '2,1'},
    {'anchor': '2,1', 'direction': 'up', 'name': '2,2'},
    {'anchor': '2,2', 'direction': 'left', 'name': '1,2'},
    {'anchor': '1,2', 'direction': 'left', 'name': '0,2'}, 
    {'anchor': '0,2', 'direction': 'down', 'name': '0,1'},
    {'anchor': '0,1', 'direction': 'right', 'name': '1,1'},
    {'anchor': '0,2', 'direction': 'up', 'name': '0,3'},
    {'anchor': '0,3', 'direction': 'up', 'name': '0,4'},
    {'anchor': '0,4', 'direction': 'right', 'name': '1,4'},
    {'anchor': '1,4', 'direction': 'down', 'name': '1,3'},
    {'anchor': '1,3', 'direction': 'right', 'name': '2,3'},
    {'anchor': '2,3', 'direction': 'up', 'name': '2,4'},
    {'anchor': '2,4', 'direction': 'right', 'name': '3,4'},
    {'anchor': '3,4', 'direction': 'down', 'name': '3,3'},
    {'anchor': '3,3', 'direction': 'down', 'name': '3,2'},
    {'anchor': '3,2', 'direction': 'right', 'name': '4,2'},
    {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
    {'anchor': '4,3', 'direction': 'up', 'name': '4,4'},
]
mazes_dict['pbcs_0'] = {'maze': Maze(*segments_pbcs_0, goal_squares='4,4'), 'action_range': 0.95}

segments_pbcs_1 = [
    {'anchor': 'origin', 'direction': 'up', 'name': '0,1'},
    {'anchor': '0,1', 'direction': 'up', 'name': '0,2'},
    {'anchor': '0,2', 'direction': 'right', 'name': '1,2'},
    {'anchor': '1,2', 'direction': 'right', 'name': '2,2'},
    {'anchor': '2,2', 'direction': 'right', 'name': '3,2'},
    {'anchor': '3,2', 'direction': 'down', 'name': '3,1'},
    {'anchor': '3,1', 'direction': 'left', 'name': '2,1'},
    {'anchor': '2,1', 'direction': 'left', 'name': '1,1'},
    {'anchor': '1,1', 'direction': 'down', 'name': '1,0'},
    {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
    {'anchor': '2,0', 'direction': 'right', 'name': '3,0'},
    {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
    {'anchor': '4,0', 'direction': 'up', 'name': '4,1'},
    {'anchor': '4,1', 'direction': 'up', 'name': '4,2'},
    {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
    {'anchor': '4,3', 'direction': 'left', 'name': '3,3'},
    {'anchor': '3,3', 'direction': 'up', 'name': '3,4'},
    {'anchor': '3,4', 'direction': 'right', 'name': '4,4'},
    {'anchor': '3,4', 'direction': 'left', 'name': '2,4'},
    {'anchor': '2,4', 'direction': 'down', 'name': '2,3'},
    {'anchor': '2,3', 'direction': 'left', 'name': '1,3'},
    {'anchor': '1,3', 'direction': 'up', 'name': '1,4'},
    {'anchor': '1,4', 'direction': 'left', 'name': '0,4'},
    {'anchor': '0,4', 'direction': 'down', 'name': '0,3'}
]
mazes_dict['pbcs_1'] = {'maze': Maze(*segments_pbcs_1, goal_squares='0,3'), 'action_range': 0.95}

mazes_dict['snail'] = {'maze': make_snail_maze(5), 'action_range': 0.95}

base_path = """/home/castanet/curiculumrl/module/results/PBCS_MED_1/SVGG/proto_env-pointmaze_alg-DDPG_herrfaab_1_4_3_1_1_seed111_tb-OCSVM_gamma_0.5_OVER_SAMPLE_annealed_E_100_PARTS_FMR_1000_ag_cu-svgd_first-True_beta-10.0_svgd_-1_oe_pa-5_succ_-4000"""
print("base_path : ", base_path)

maze_type = 'pbcs_1'

# Get particles

particles = {}
goal_particles = {}
target_distrib = {}
prior_distrib = {}

max_timestep = 200000
min_timestep = 0

timestep_list_part = [folder for folder in os.listdir(base_path+'/particles') if folder.isdigit()]
timestep_dict_part = {}
for time in timestep_list_part:
    timestep_dict_part[int(time)] = time

timestep_list_part = [int(t) for t in timestep_list_part]
timestep_list_part.sort()

scale = 1000
episode_length = 50
do_filter_unique = True # For last_bgs
skip_every_part = 2


folder_name = "particles"
particles = extract_success_data(base_path+'/particles', timestep_dict_part, timestep_list_part, min_timestep, max_timestep, folder_name, episode_length, scale, skip_every_part, filter_unique=False)

folder_name = "distrib"
target_distrib = extract_success_data(base_path+'/particles', timestep_dict_part, timestep_list_part, min_timestep, max_timestep, folder_name, episode_length, scale, skip_every_part, filter_unique=False)


# Get coverage
timestep_list_cov = [folder for folder in os.listdir(base_path+'/grid_eval') if folder.isdigit()]
timestep_dict_cov = {}
for time in timestep_list_cov:
    timestep_dict_cov[int(time)] = time

    
timestep_list_cov = [int(t) for t in timestep_list_cov]
timestep_list_cov.sort()

scale = 1000
episode_length = 50
do_filter_unique = True # For last_bgs
skip_every_cov = 1

folder_name = "success"
grid_succ = extract_success_data(base_path+'/grid_eval', timestep_dict_cov, timestep_list_cov, min_timestep, max_timestep, folder_name, episode_length, scale, skip_every_cov, filter_unique=False)



# Grid array
h = 0.2
x_min, x_max = -2.5, 11.5
y_min, y_max = -2.5, 11.5
xx,yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))


m = mazes_dict[maze_type]['maze']
locs = np.array(list(m._locs))
d_min = locs.min()-0.5
d_max = locs.max()+0.5

h = 1.1
x_min, x_max = d_min, d_max
y_min, y_max = d_min, d_max
xx_0,yy_0 = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

# Gif

gif_path = base_path+"/gif"
t_init = 0
filenames = []

if not os.path.exists(gif_path):
    print("Creating gif folder ...\n")
    os.mkdir(gif_path)
else:
    for (_, _, f) in os.walk(gif_path):
        if len(f) == 0:
            print("\nGif folder already exist but is empty\n")
            break
        else:
            t_init = max([int(f.split(".")[0]) for f in f])
            filenames = f
            filenames.sort(key=lambda x: int(x.split('.')[0]))
            filenames = [gif_path+'/'+ f for f in filenames]
            print(f'\nGif folder already exist and contains image up to t = {t_init}\n')
        break


print(20*'-')
print("init : ",t_init)
print(20*'-'+'\n')

fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(13,7))

T = len(particles)

cov_ind = -1

for t in range(t_init,T):
    if t % 50 == 0:
        print(f'image : {t}/{T}')
        print(10*'-'+'\n')
    time_part = timestep_list_part[::skip_every_part][t]
    mazes_dict[maze_type]['maze'].plot(axs[0])
    mazes_dict[maze_type]['maze'].plot(axs[1])
    axs[0].axis('off')
    axs[1].axis('off')

    part = particles[t]
        
    # Particles
    axs[0].contourf(xx, yy, target_distrib[t].reshape(xx.shape), cmap='viridis', alpha=0.8) #,levels=20)
    axs[0].scatter(part[:,0], part[:,1], c='r', marker='+')

    axs[0].set_title("Step {}".format(time_part), fontweight="bold")
    
    plot_cov = True
    if plot_cov:
        # Coverage
        if time_part > timestep_list_cov[::skip_every_cov][cov_ind+1]:
            cov_ind+=1

        if cov_ind == -1:
            cov = np.zeros(grid_succ[0].shape)
        else:
            cov = grid_succ[cov_ind]

        nb_samples = 10
        cov = cov.reshape(xx_0.shape[0]**2,nb_samples,-1)
        cov = cov.mean(axis=1)

        sc = axs[1].imshow(cov.reshape(xx_0.shape[0],-1).swapaxes(0,1),origin='lower',extent=(d_min,d_max,d_min,d_max),vmin=0,vmax=1, cmap='RdBu', alpha=0.5)
        axs[1].set_title("Cov. {} %".format(round(cov.mean(),2)), fontweight="bold")
        
        #fig.colorbar(sc, ax=axs[1])
        for (j,i),label in np.ndenumerate(cov.reshape(xx_0.shape[0],-1).swapaxes(0,1)):
            axs[1].text(i,j,round(label,2),ha='center',va='center')
    
    filename = base_path+f'/gif/{t}.png'
    filenames.append(filename)
    plt.savefig(filename)

    plt.cla()


print("Build Gif")

# build gif
with imageio.get_writer(base_path+f'/mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)