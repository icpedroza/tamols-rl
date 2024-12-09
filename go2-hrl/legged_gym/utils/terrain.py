import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy.ndimage import binary_dilation

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            print("curriculum")
            self.curiculum()
        elif cfg.selected:
            print("selected terrain")
            self.selected_terrain()
        else:  
            print("randomized terrain")  
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
                self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
        # self.vertices, self.triangles, self.x_edge_mask = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
        #                                                                                     self.cfg.horizontal_scale,
        #                                                                                     self.cfg.vertical_scale,
        #                                                                                     self.cfg.slope_treshold)
        # half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
        # structure = np.ones((half_edge_width*2+1, 1))
        # self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

        assert np.sum(cfg.terrain_proportions) == 1.0, "Terrain proportions must sum to 1.0"
        
    def randomized_terrain(self):
        print("Generating randomized terrain...")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            print(f"Sub-terrain {k}: min height = {np.min(terrain.height_field_raw)}, max height = {np.max(terrain.height_field_raw)}")  # Debug print
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            # eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        # if choice < self.proportions[0]:
        #     if choice < self.proportions[0]/ 2:
        #         slope *= -1
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # elif choice < self.proportions[1]:
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        # elif choice < self.proportions[3]:
        #     if choice<self.proportions[2]:
        #         step_height *= -1
        #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        # elif choice < self.proportions[4]:
        #     num_rectangles = 20
        #     rectangle_min_size = 1.
        #     rectangle_max_size = 2.
        #     terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        # elif choice < self.proportions[5]:
        #     terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        # elif choice < self.proportions[6]:
        #     gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        # else:
        #     pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        # climbing_course_terrain(terrain, 
        #                   column_height=1.0 + difficulty * 0.5,  # Columns get higher with difficulty
        #                   beam_height=0.5 + difficulty * 0.25,   # Beams get higher with difficulty
        #                   platform_size=3.)
        
        # stepping_stones_terrain(terrain, 
        #                         stone_size = 0.1 + difficulty, 
        #                         stone_distance = 0.4 + (0.1 * difficulty),
        #                         max_height = 0.5 + difficulty)

        # demo_terrain(terrain)
        # parkour_terrain(terrain)
        # complex_parkour_terrain(terrain, difficulty)
        column_field_terrain(terrain, difficulty) #TODO: if you want flat terrain uncomment this
        # varied_column_field_terrain(terrain, difficulty)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -4000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def climbing_course_terrain(terrain, column_height=1.0, beam_height=0.5, platform_size=3.):
    """Creates a terrain with columns and connecting beams for climbing practice
    
    Args:
        terrain: SubTerrain object
        column_height: Height of the columns
        beam_height: Height of the connecting beams
        platform_size: Size of the starting/ending platforms
    """
    # Clear the terrain first
    terrain.height_field_raw.fill(0)
    
    # Starting platform
    start_x = int(platform_size / terrain.horizontal_scale)
    terrain.height_field_raw[:start_x, :] = 0
    
    # Calculate center positions
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    
    # Convert heights to height field units
    column_height_units = int(column_height / terrain.vertical_scale)
    beam_height_units = int(beam_height / terrain.vertical_scale)
    
    # Create columns
    column_positions = [
        (center_x - 20, center_y - 15),
        (center_x + 20, center_y - 15),
        (center_x - 20, center_y + 15),
        (center_x + 20, center_y + 15),
        (center_x, center_y)
    ]
    
    column_width = int(1.0 / terrain.horizontal_scale)  # 1m wide columns
    
    # Place columns
    for col_x, col_y in column_positions:
        x1, x2 = col_x - column_width//2, col_x + column_width//2
        y1, y2 = col_y - column_width//2, col_y + column_width//2
        terrain.height_field_raw[x1:x2, y1:y2] = column_height_units
    
    # Create connecting beams
    beam_width = int(0.5 / terrain.horizontal_scale)  # 0.5m wide beams
    
    # Horizontal beams
    for i in range(len(column_positions)-1):
        for j in range(i+1, len(column_positions)):
            x1, y1 = column_positions[i]
            x2, y2 = column_positions[j]
            
            # Draw beam
            if abs(x1 - x2) > abs(y1 - y2):  # Horizontal beam
                beam_y = min(y1, y2) + abs(y2 - y1)//2
                x_start, x_end = min(x1, x2), max(x1, x2)
                terrain.height_field_raw[x_start:x_end, 
                                      beam_y-beam_width//2:beam_y+beam_width//2] = beam_height_units
            else:  # Vertical beam
                beam_x = min(x1, x2) + abs(x2 - x1)//2
                y_start, y_end = min(y1, y2), max(y1, y2)
                terrain.height_field_raw[beam_x-beam_width//2:beam_x+beam_width//2,
                                      y_start:y_end] = beam_height_units
                
def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    # goals[5] = [platform_length+slope_depth/2, left_y]
    # heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    # terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    # platform_length += slope_depth + gap_size
    # goals[6] = [platform_length+slope_depth/2, right_y]
    # heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    # terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale
    return terrain

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    
    # Ensure minimum stone length
    min_stone_len = 0.3  # minimum 30cm
    stone_len = max(min_stone_len, np.random.uniform(min_stone_len, stone_len))
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = max(2, round(stone_len / terrain.horizontal_scale))  # Ensure at least 2 pixels
    
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = max(2, round(stone_width / terrain.horizontal_scale))  # Ensure at least 2 pixels
    last_stone_len = max(2, round(last_stone_len / terrain.horizontal_scale))

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    dis_z = np.random.randint(dis_z_min, dis_z_max)
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            x_start = max(0, dis_x-last_stone_len//2)
            x_end = min(terrain.height_field_raw.shape[0], dis_x+last_stone_len//2)
            y_start = max(0, dis_y-stone_width//2)
            y_end = min(terrain.height_field_raw.shape[1], dis_y+stone_width//2)
            
            if x_end > x_start and y_end > y_start:
                terrain.height_field_raw[x_start:x_end, y_start:y_end] = heights[:x_end-x_start, :y_end-y_start].astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            x_start = max(0, dis_x-stone_len//2)
            x_end = min(terrain.height_field_raw.shape[0], dis_x+stone_len//2)
            y_start = max(0, dis_y-stone_width//2)
            y_end = min(terrain.height_field_raw.shape[1], dis_y+stone_width//2)
            
            if x_end > x_start and y_end > y_start:
                terrain.height_field_raw[x_start:x_end, y_start:y_end] = heights[:x_end-x_start, :y_end-y_start].astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]
        left_right_flag = 1 - left_right_flag
    
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = min(terrain.height_field_raw.shape[0]-1, 
                             dis_x + last_stone_len // 2 + round(0.05 / terrain.horizontal_scale))
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = max(1, int(pad_width / terrain.horizontal_scale))
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
    return terrain

def climbing_course_terrain2(terrain, column_height=1.0, beam_height=0.5, platform_size=3.):
    """Creates a terrain with stairs leading to columns and connecting beams
    
    Args:
        terrain: SubTerrain object
        column_height: Height of the columns
        beam_height: Height of the connecting beams
        platform_size: Size of the starting/ending platforms
    """
    # Clear the terrain first
    terrain.height_field_raw.fill(0)
    
    # Calculate center positions
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    
    # Convert heights to height field units
    column_height_units = int(column_height / terrain.vertical_scale)
    beam_height_units = int(beam_height / terrain.vertical_scale)
    
    # Create stairs parameters
    num_steps = 8
    step_length = int(0.4 / terrain.horizontal_scale)  # 40cm per step
    step_width = int(2.0 / terrain.horizontal_scale)   # 2m wide stairs
    step_height = beam_height_units // num_steps       # Divide beam height into equal steps
    
    # Create two sets of stairs on either side
    for side in [-1, 1]:
        stair_center_y = center_y + (side * int(10 / terrain.horizontal_scale))
        
        # Create each step
        for step in range(num_steps):
            step_x_start = center_x - (num_steps - step) * step_length
            step_x_end = step_x_start + step_length
            step_y_start = stair_center_y - step_width//2
            step_y_end = stair_center_y + step_width//2
            
            current_height = (step + 1) * step_height
            terrain.height_field_raw[step_x_start:step_x_end, 
                                   step_y_start:step_y_end] = current_height
    
    # Create columns
    column_positions = [
        (center_x - 20, center_y - 15),
        (center_x + 20, center_y - 15),
        (center_x - 20, center_y + 15),
        (center_x + 20, center_y + 15),
        (center_x, center_y)
    ]
    
    column_width = int(1.0 / terrain.horizontal_scale)  # 1m wide columns
    
    # Place columns
    for col_x, col_y in column_positions:
        x1, x2 = col_x - column_width//2, col_x + column_width//2
        y1, y2 = col_y - column_width//2, col_y + column_width//2
        terrain.height_field_raw[x1:x2, y1:y2] = column_height_units
    
    # Create connecting beams
    beam_width = int(0.5 / terrain.horizontal_scale)  # 0.5m wide beams
    
    # Create landing platforms at the top of the stairs
    platform_length = int(2.0 / terrain.horizontal_scale)
    platform_width = int(2.0 / terrain.horizontal_scale)
    
    for side in [-1, 1]:
        platform_y = center_y + (side * int(10 / terrain.horizontal_scale))
        platform_x = center_x
        
        terrain.height_field_raw[platform_x:platform_x+platform_length,
                               platform_y-platform_width//2:platform_y+platform_width//2] = beam_height_units
    
    # Horizontal beams connecting everything
    for i in range(len(column_positions)-1):
        for j in range(i+1, len(column_positions)):
            x1, y1 = column_positions[i]
            x2, y2 = column_positions[j]
            
            # Draw beam
            if abs(x1 - x2) > abs(y1 - y2):  # Horizontal beam
                beam_y = min(y1, y2) + abs(y2 - y1)//2
                x_start, x_end = min(x1, x2), max(x1, x2)
                terrain.height_field_raw[x_start:x_end, 
                                      beam_y-beam_width//2:beam_y+beam_width//2] = beam_height_units
            else:  # Vertical beam
                beam_x = min(x1, x2) + abs(x2 - x1)//2
                y_start, y_end = min(y1, y2), max(y1, y2)
                terrain.height_field_raw[beam_x-beam_width//2:beam_x+beam_width//2,
                                      y_start:y_end] = beam_height_units
                
def complex_parkour_terrain(terrain, difficulty=1.0):
    """Creates a complex parkour terrain with various obstacles similar to the reference image
    
    Args:
        terrain: SubTerrain object
        difficulty: Scales the height and gap distances of obstacles
    """
    # Clear the terrain first
    terrain.height_field_raw.fill(-100)  # Create a pit/void
    
    # Get terrain dimensions
    length = terrain.length
    width = terrain.width
    
    # Calculate center line
    center_x = length // 2
    center_y = width // 2
    
    # Convert common measurements to terrain units
    def m_to_terrain(meters, is_height=False):
        if is_height:
            return int(meters / terrain.vertical_scale)
        return int(meters / terrain.horizontal_scale)
    
    # Scale all dimensions down to fit in the terrain
    scale_factor = 0.15  # Adjust this to make obstacles smaller/larger
    
    # Basic platform parameters
    platform_height = m_to_terrain(0.5 * scale_factor, is_height=True)
    narrow_width = max(2, m_to_terrain(0.8 * scale_factor))
    standard_width = max(3, m_to_terrain(1.2 * scale_factor))
    
    # Create starting platform
    start_length = min(m_to_terrain(2.0 * scale_factor), length // 8)
    y_start = max(0, center_y - standard_width)
    y_end = min(width, center_y + standard_width)
    terrain.height_field_raw[:start_length, y_start:y_end] = 0
    
    current_x = start_length
    
    # 1. Initial rolling terrain with cylinders
    num_cylinders = 2
    cylinder_radius = max(2, m_to_terrain(0.3 * scale_factor))
    cylinder_spacing = max(3, m_to_terrain(0.8 * scale_factor))
    
    for i in range(num_cylinders):
        if current_x + 2*cylinder_radius >= length:
            break
        x_center = current_x + cylinder_radius + i*cylinder_spacing
        for dx in range(-cylinder_radius, cylinder_radius+1):
            if 0 <= x_center+dx < length:
                height = int(np.sqrt(max(0, cylinder_radius**2 - dx**2)))
                terrain.height_field_raw[x_center+dx, y_start:y_end] = height
    
    current_x = min(current_x + num_cylinders*cylinder_spacing + m_to_terrain(1.0 * scale_factor), length-1)
    
    # 2. Angled platforms section
    num_angles = 2
    platform_length = min(m_to_terrain(1.5 * scale_factor), (length - current_x) // (num_angles + 1))
    angle_height = m_to_terrain(0.4 * scale_factor, is_height=True)
    
    for i in range(num_angles):
        if current_x + platform_length >= length:
            break
        start_height = i * angle_height
        end_height = (i+1) * angle_height
        x_start = current_x + i*platform_length
        x_end = min(x_start + platform_length, length)
        
        # Create angled platform
        for x in range(x_start, x_end):
            progress = (x - x_start) / platform_length
            height = int(start_height + progress * (end_height - start_height))
            y_narrow_start = max(0, center_y - narrow_width)
            y_narrow_end = min(width, center_y + narrow_width)
            terrain.height_field_raw[x, y_narrow_start:y_narrow_end] = height
    
    current_x = min(current_x + num_angles*platform_length + m_to_terrain(1.0 * scale_factor), length-1)
    
    # 3. Vertical posts section
    num_posts = 3
    post_width = max(2, m_to_terrain(0.4 * scale_factor))
    post_spacing = max(3, m_to_terrain(0.8 * scale_factor))
    post_heights = [m_to_terrain(h * scale_factor, is_height=True) for h in [1.0, 1.2, 1.0]]
    
    for i in range(num_posts):
        if current_x + post_width >= length:
            break
        x_center = current_x + i*post_spacing
        x_end = min(x_center + post_width, length)
        y_post_start = max(0, center_y - post_width//2)
        y_post_end = min(width, center_y + post_width//2)
        terrain.height_field_raw[x_center:x_end, y_post_start:y_post_end] = post_heights[i]
    
    current_x = min(current_x + num_posts*post_spacing + m_to_terrain(1.0 * scale_factor), length-1)
    
    # 4. Final platform
    if current_x < length - 1:
        end_length = min(m_to_terrain(2.0 * scale_factor), length - current_x)
        terrain.height_field_raw[current_x:current_x+end_length, y_start:y_end] = 0
    
    # Add safety walls on the sides
    wall_height = m_to_terrain(1.0 * scale_factor, is_height=True)
    wall_width = max(1, m_to_terrain(0.5 * scale_factor))
    terrain.height_field_raw[:, :wall_width] = wall_height
    terrain.height_field_raw[:, -wall_width:] = wall_height
    
    return terrain

def column_field_terrain(terrain, difficulty=1.0):
    """Creates a terrain with a dense field of small, closely spaced columns that rise from the pit to ground level
    
    Args:
        terrain: SubTerrain object
        difficulty: Scales the gap distances
    """
    # Set pit depth and matching column height
    pit_depth = -100  # Deep pit
    terrain.height_field_raw.fill(pit_depth)
    
    # Get terrain dimensions
    length = terrain.length
    width = terrain.width
    
    def m_to_terrain(meters, is_height=False):
        if is_height:
            return int(meters / terrain.vertical_scale)
        return int(meters / terrain.horizontal_scale)
    
    # Column parameters - much smaller and closer together
    column_width = m_to_terrain(0.35)  # 15cm wide columns (reduced from 50cm)
    column_spacing = m_to_terrain(0.11)  # 5cm between columns (reduced from 40cm)
    column_height = 0  # Columns reach ground level (height of 0)
    
    # Calculate number of columns that can fit in each dimension
    num_cols_x = (length - column_width) // (column_width + column_spacing)
    num_cols_y = (width - column_width) // (column_width + column_spacing)
    
    # Create starting platform
    start_platform_length = m_to_terrain(1.0)
    terrain.height_field_raw[:start_platform_length, :] = column_height
    
    # Create columns in a grid pattern
    for i in range(int(num_cols_x)):
        for j in range(int(num_cols_y)):
            # Calculate column position
            x_start = start_platform_length + i * (column_width + column_spacing)
            x_end = x_start + column_width
            y_start = j * (column_width + column_spacing)
            y_end = y_start + column_width
            
            # Place column if within bounds
            if x_end < length and y_end < width:
                # Add slight random variation to column height to make it more interesting
                height_variation = np.random.randint(-m_to_terrain(0.05, is_height=True), 
                                                   m_to_terrain(0.05, is_height=True))
                terrain.height_field_raw[x_start:x_end, y_start:y_end] = column_height + height_variation
    
    # Create end platform
    end_platform_length = m_to_terrain(1.0)
    if length - end_platform_length > 0:
        terrain.height_field_raw[-end_platform_length:, :] = column_height
    
    return terrain

def varied_column_field_terrain(terrain, difficulty=1.0):
    """Creates a terrain with a field of closely spaced columns with slight height variations
    
    Args:
        terrain: SubTerrain object
        difficulty: Scales the gap distances
    """
    # Set pit depth
    pit_depth = -40000  
    terrain.height_field_raw.fill(pit_depth)
    
    # Get terrain dimensions
    length = terrain.length
    width = terrain.width
    
    def m_to_terrain(meters, is_height=False):
        if is_height:
            return int(meters / terrain.vertical_scale)
        return int(meters / terrain.horizontal_scale)
    
    # Column parameters
    column_width = m_to_terrain(0.35)  # 35cm wide columns
    column_spacing = m_to_terrain(0.25)  # 25cm between columns
    base_height = 0  # Base ground level height
    
    # Height variation parameters - much more variation now
    min_height = m_to_terrain(-0.3, is_height=True)  # -30cm variation
    max_height = m_to_terrain(0.3, is_height=True)   # +30cm variation
    
    # Calculate number of columns that can fit in each dimension
    num_cols_x = (length - column_width) // (column_width + column_spacing)
    num_cols_y = (width - column_width) // (column_width + column_spacing)
    
    # Create starting platform
    start_platform_length = m_to_terrain(1.0)
    terrain.height_field_raw[:start_platform_length, :] = base_height
    
    # Create columns in a grid pattern
    # Use a mix of random heights and some patterns
    for i in range(int(num_cols_x)):
        for j in range(int(num_cols_y)):
            # Calculate column position
            x_start = start_platform_length + i * (column_width + column_spacing)
            x_end = x_start + column_width
            y_start = j * (column_width + column_spacing)
            y_end = y_start + column_width
            
            # Generate varied heights using different patterns
            pattern_choice = np.random.random()
            
            if pattern_choice < 0.4:  # 40% completely random
                current_height = base_height + np.random.randint(min_height, max_height)
            elif pattern_choice < 0.6:  # 20% wave pattern
                current_height = base_height + int(max_height * 0.7 * np.sin(i * 0.5) * np.cos(j * 0.5))
            elif pattern_choice < 0.8:  # 20% distance-based
                dist_from_center = np.sqrt((i - num_cols_x/2)**2 + (j - num_cols_y/2)**2)
                current_height = base_height + int(max_height * 0.8 * (1 - dist_from_center/(num_cols_x/2)))
            else:  # 20% checkerboard-like pattern
                current_height = base_height + (max_height if (i + j) % 2 == 0 else min_height)
            
            # Add small random noise to all patterns
            current_height += np.random.randint(-m_to_terrain(0.05, is_height=True), 
                                              m_to_terrain(0.05, is_height=True))
            
            # Place column if within bounds
            if x_end < length and y_end < width:
                terrain.height_field_raw[x_start:x_end, y_start:y_end] = current_height
    
    # Create end platform
    end_platform_length = m_to_terrain(1.0)
    if length - end_platform_length > 0:
        terrain.height_field_raw[-end_platform_length:, :] = base_height
    
    return terrain

def round_varied_column_field_terrain(terrain, difficulty=1.0):
    """Creates a terrain with a field of closely spaced round columns with height variations
    
    Args:
        terrain: SubTerrain object
        difficulty: Scales the gap distances
    """
    # Set pit depth
    pit_depth = -40000  
    terrain.height_field_raw.fill(pit_depth)
    
    # Get terrain dimensions
    length = terrain.length
    width = terrain.width
    
    def m_to_terrain(meters, is_height=False):
        if is_height:
            return int(meters / terrain.vertical_scale)
        return int(meters / terrain.horizontal_scale)
    
    # Column parameters
    column_radius = m_to_terrain(0.175)  # 35cm diameter columns
    column_spacing = m_to_terrain(0.25)  # 25cm between columns
    base_height = 0  # Base ground level height
    
    # Height variation parameters
    min_height = m_to_terrain(-0.3, is_height=True)  # -30cm variation
    max_height = m_to_terrain(0.3, is_height=True)   # +30cm variation
    
    # Calculate number of columns that can fit in each dimension
    num_cols_x = (length - 2*column_radius) // (2*column_radius + column_spacing)
    num_cols_y = (width - 2*column_radius) // (2*column_radius + column_spacing)
    
    # Create starting platform
    start_platform_length = m_to_terrain(1.0)
    terrain.height_field_raw[:start_platform_length, :] = base_height
    
    # Create circular columns in a grid pattern
    for i in range(int(num_cols_x)):
        for j in range(int(num_cols_y)):
            # Calculate column center position
            center_x = start_platform_length + column_radius + i * (2*column_radius + column_spacing)
            center_y = column_radius + j * (2*column_radius + column_spacing)
            
            # Generate varied heights using different patterns
            pattern_choice = np.random.random()
            
            if pattern_choice < 0.4:  # 40% completely random
                current_height = base_height + np.random.randint(min_height, max_height)
            elif pattern_choice < 0.6:  # 20% wave pattern
                current_height = base_height + int(max_height * 0.7 * np.sin(i * 0.5) * np.cos(j * 0.5))
            elif pattern_choice < 0.8:  # 20% distance-based
                dist_from_center = np.sqrt((i - num_cols_x/2)**2 + (j - num_cols_y/2)**2)
                current_height = base_height + int(max_height * 0.8 * (1 - dist_from_center/(num_cols_x/2)))
            else:  # 20% checkerboard-like pattern
                current_height = base_height + (max_height if (i + j) % 2 == 0 else min_height)
            
            # Add small random noise
            current_height += np.random.randint(-m_to_terrain(0.05, is_height=True), 
                                              m_to_terrain(0.05, is_height=True))
            
            # Create circular column by checking distance from center
            for dx in range(-column_radius, column_radius + 1):
                for dy in range(-column_radius, column_radius + 1):
                    # Calculate distance from column center
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # If within radius, set height
                    if distance <= column_radius:
                        x = center_x + dx
                        y = center_y + dy
                        
                        # Check bounds
                        if 0 <= x < length and 0 <= y < width:
                            terrain.height_field_raw[int(x), int(y)] = current_height
    
    # Create end platform
    end_platform_length = m_to_terrain(1.0)
    if length - end_platform_length > 0:
        terrain.height_field_raw[-end_platform_length:, :] = base_height
    
    return terrain