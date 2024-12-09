import numpy as np


def get_flat_heightmap(tmls):
    return np.zeros((tmls.map_size, tmls.map_size))


def get_platform_heightmap(tmls):
    grid_size = tmls.map_size  # 50 x 0.04 = 2 meters
    elevation_map = np.zeros((grid_size, grid_size))

    platform_size_x = grid_size // 2
    platform_size_y = grid_size // 2
    platform_height = 0.05

    start_x = (grid_size - platform_size_x) // 2
    end_x = start_x + platform_size_x
    start_y = (grid_size - platform_size_y) // 2
    end_y = start_y + platform_size_y

    # Add the raised square platform
    elevation_map[start_x:end_x, start_y:end_y] = platform_height

    return elevation_map  


def get_random_rough_heightmap(tmls):
    grid_size = tmls.map_size

    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    print(f"Random seed used: {seed}")

    elevation_map = np.random.rand(grid_size, grid_size) * 0.05

    # Smooth the heightmap by averaging with neighbors
    for _ in range(1):  # Number of smoothing iterations
        elevation_map = (np.roll(elevation_map, 1, axis=0) + np.roll(elevation_map, -1, axis=0) +
                         np.roll(elevation_map, 1, axis=1) + np.roll(elevation_map, -1, axis=1) +
                         elevation_map) / 5.0
    return elevation_map


def get_heightmap_with_holes(tmls):
    grid_size = tmls.map_size
    elevation_map = np.random.rand(grid_size, grid_size) * 0.05
    drop_height = 0.75

    # Smooth the heightmap by averaging with neighbors
    for _ in range(1):  # Number of smoothing iterations
        elevation_map = (np.roll(elevation_map, 1, axis=0) + np.roll(elevation_map, -1, axis=0) +
                         np.roll(elevation_map, 1, axis=1) + np.roll(elevation_map, -1, axis=1) +
                         elevation_map) / 5.0

    # Add random holes
    num_holes = np.random.randint(10, 20)  # Random number of holes
    for _ in range(num_holes):
        hole_x = np.random.randint(0, grid_size)
        hole_y = np.random.randint(0, grid_size)
        elevation_map[hole_x, hole_y] = -drop_height

    return elevation_map

def get_heightmap_stairs(tmls):
    grid_size = tmls.map_size
    elevation_map = np.zeros((grid_size, grid_size))
    step_height = 0.05
    middle_index = 3*(grid_size // 4)
    for i in range(middle_index, grid_size, 4):
        elevation_map[i:i+4, :] = step_height * ((i - middle_index) // 4 + 1)
    return elevation_map

