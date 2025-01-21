import simulation
import dearpygui.dearpygui as dpg

# Constants
block_size = 10 # 10px * 10px
rows = 23
columns = 23
arena: simulation.Arena = simulation.Arena()
width = columns * block_size
height = rows * block_size

def create_texture(n: int, m: int) -> list[float]:
    texture_data: list[float] = []
    for _ in range(0, n * m):
        texture_data.append(255 / 255)
        texture_data.append(0)
        texture_data.append(255 / 255)
        texture_data.append(255 / 255)
    return texture_data

# Updates the grid text with the current state of the arena
def update_grid(tag: str | int = 'Environment'):
    global arena
    texture_data: list[float] = []
    _map: dict[int, list[float]] = {
        simulation.Arena.EMPTY: [0, 0, 0, 0], # Black
        simulation.Arena.WALL: [0, 0, 255 / 255, 255 / 255], # Blue
        simulation.Arena.PLAYER: [255/ 255, 255 / 255, 0, 255/ 255], # Yellow
        simulation.Arena.GOAL: [0, 255 / 255, 0, 255 / 255], # Green
    }

    for y in range(arena.m): # Every Y coordinate
        for _ in range(block_size): # Every Y block
            for x in range(arena.n): # Every X coordinate
                for _ in range(block_size): # Every X block
                    # RGBA pixel format
                    texture_data.extend(_map[arena.grid[y][x]]) # Pixel value
    dpg.set_value(item=tag, value=texture_data)

def move(sender, app_data, user_data: simulation.Direction):
    arena.move(simulation.Direction(user_data))
    if arena.on_goal():
        arena.set_goal()
    update_grid()

if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title='Pacman [float]', width=800, height=600)
    dpg.setup_dearpygui()

    arena_texture: list[float] = create_texture(rows * block_size, columns * block_size)
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width=width, height=height, default_value=arena_texture, tag='Environment')
    update_grid()

    with dpg.window(tag="Pacman"):
        dpg.add_image("Environment", width=width, height=height)

    with dpg.handler_registry():
        dpg.add_key_press_handler(key = dpg.mvKey_W, user_data = simulation.Direction.UP, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_A, user_data = simulation.Direction.LEFT, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_S, user_data = simulation.Direction.DOWN, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_D, user_data = simulation.Direction.RIGHT, callback=move)

    dpg.show_viewport()
    dpg.set_primary_window("Pacman", True)
    
    while dpg.is_dearpygui_running():
        #update_grid()
        jobs = dpg.get_callback_queue()
        dpg.run_callbacks(jobs)
        
        dpg.render_dearpygui_frame()
    dpg.destroy_context()