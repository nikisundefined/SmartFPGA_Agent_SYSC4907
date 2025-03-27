import smart_agent.simulation as simulation
import dearpygui.dearpygui as dpg
import time
import numpy as np
import logging
import smart_agent

log: logging.Logger = logging.getLogger('smart_agent.gui')

# Gives a block view on the GUI backing texture for the given coordinates
def block(x: int, y: int, block_size: int = 10) -> np.ndarray:
    grid: np.ndarray = smart_agent.gvar.arena.grid
    pass

# Updates the backing texture based on the shared arena instance
def update() -> None:
    previous_grid: np.ndarray = smart_agent.gvar.previous_grid
    texture: np.ndarray = smart_agent.gvar.texture
    grid: np.ndarray = smart_agent.gvar.arena.grid
    arena: simulation.Arena = smart_agent.gvar.arena
    block_size: int = smart_agent.gvar.block_size

    COLOR_MAP = {
        0: np.array([0, 0, 0], dtype=np.float32),       # Empty (Black)
        1: np.array([0, 0, 1], dtype=np.float32),       # Wall (Blue)
        2: np.array([1, 1, 0], dtype=np.float32),       # Player (Yellow)
        3: np.array([0, 1, 0], dtype=np.float32),       # Goal (Green)
        'path': np.array([1, 0, 0], dtype=np.float32)   # Path (Red)
    }
    
    height, width = grid.shape
    path = arena.distance()  # Get the computed path

    # Initialize previous_grid if it's the first update
    if previous_grid is None:
        previous_grid = np.copy(grid)  # Store a copy of the grid initially

    # Iterate through the grid and update the corresponding texture region
    for y in range(height):
        for x in range(width):
            point = simulation.Point(x, y)

            # Check if the grid cell has changed
            # if grid[y, x] != previous_grid[y, x] or point in path or previous_grid[y, x] == simulation.Arena.PLAYER:
            color = COLOR_MAP.get(int(grid[y, x]), COLOR_MAP[0])  # Default to black if unknown

            # Assign the color to the corresponding block in the texture
            for oy in range(block_size):
                for ox in range(block_size):
                    if point in path and grid[y, x] == 0 and (2 < ox < 8) and (2 < oy < 8):
                        texture[y * block_size + oy, x * block_size + ox] = COLOR_MAP['path']
                    else:
                        texture[y * block_size + oy, x * block_size + ox] = color
                            
    # Update previous_grid for the next call
    previous_grid = np.copy(grid)

def move(sender, app_data, user_data: simulation.Direction):
    arena = smart_agent.cvar.arena
    old = arena.player.point.copy()

    arena.move(simulation.Direction(user_data))
    if arena.on_goal():
        arena.set_goal()
        dpg.set_value('score', f'Score: {arena.player.score}')
    update()
    try:
        print(f'Current: {arena.player} [{arena.grid[arena.player.y, arena.player.x]}]\nOld: {old} [{arena.grid[old.y, old.x]}]')
        print(arena.detection())
    except:
        print(f"DEBUG: {arena.player}")
        print(f"DEBUG: {arena}")

def inhibit(sender, app_data, user_data: bool):
    smart_agent.cvar.learning = False

def create_gui(arena: simulation.Arena | None = None, block_size: int = 10):
    # If the arena is not given load attempt to load it from shared memory
    if arena is None:
        arena = smart_agent.gvar.arena
    rows: int = arena.n
    columns: int = arena.m
    # Compute the viewport size with extra padding
    VIEWPORT_WIDTH: int = round((rows * block_size) / 100.0 + 0.5) * 100
    VIEWPORT_HEIGHT: int = round((columns * block_size) / 100.0 + 0.5) * 100

    # Texture size of the grid representation
    texture_width: int = rows * block_size
    texture_height: int = columns * block_size

    dpg.create_context()
    dpg.create_viewport(title='Pacman [float]', width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT, vsync=False)
    dpg.setup_dearpygui()

    # Create the grid texture and update it based on the arena state
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=texture_width, height=texture_height, default_value=smart_agent.gvar.texture, format=dpg.mvFormat_Float_rgb, tag='Environment')
    update()

    # Add all elements that will be displayed to the GUI
    with dpg.window(tag="Pacman"):
        hori_offset = (dpg.get_viewport_width() - texture_width) / 2
        vert_offset = (dpg.get_viewport_height() - texture_height) / 2
        dpg.add_image("Environment", width=texture_width, height=texture_height, pos=[hori_offset, vert_offset])
        dpg.add_text(f'Score: {arena.player.score}', tag='score')
        dpg.add_text('Time: 0.0', tag='time')
        dpg.add_text(f"{smart_agent.gvar.seed}", tag='seed')
        dpg.add_text(f"Is learning: {smart_agent.cvar.learning}", tag='learning')
    
    dpg.add_value_registry(tag='value_registry')
    dpg.add_float_value(tag='timer', parent='value_registry')

    # Add key handlers for moving the agent through the gui
    with dpg.handler_registry():
        dpg.add_key_press_handler(key = dpg.mvKey_W, user_data = simulation.Direction.UP, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_A, user_data = simulation.Direction.LEFT, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_S, user_data = simulation.Direction.DOWN, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_D, user_data = simulation.Direction.RIGHT, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_L, callback=inhibit)

def display_gui():
    dpg.show_viewport()
    dpg.set_value('timer', time.time())
    dpg.set_primary_window("Pacman", True)

def update_text():
    cvar = smart_agent.cvar
    gvar = smart_agent.gvar

    # Always update the score text box and reposition accordingly
    dpg.set_value('score', f'Score: {cvar.arena.player.score}')
    txt_rect = dpg.get_item_rect_size('score')
    dpg.set_item_pos('score', [dpg.get_viewport_width()/2-txt_rect[0]/2, 0])

    # Update the value and position of the seed for this run
    dpg.set_value('seed', f'{gvar.seed}')
    dpg.set_item_pos('seed', [dpg.get_viewport_width()/2-dpg.get_item_rect_size('seed')[0]/2, 265])

    # Update the time elasped within the simulation and its position
    dpg.set_value('time', f"Time {round(gvar.sim_time, 1)}")
    tim_rect = dpg.get_item_rect_size('time')
    dpg.set_item_pos('time', [dpg.get_viewport_width()/2-tim_rect[0]/2, txt_rect[1]])

    # Update the learning boolean
    dpg.set_value('learning', f"Is learning: {cvar.learning}")
    learn_rect = dpg.get_item_rect_size('learning')
    dpg.set_item_pos('learning', [dpg.get_viewport_width()/2-learn_rect[0]/2, 280])

if __name__ == "__main__":
    import time
    # Constants
    block_size = 10 # 10px * 10px
    rows = 23
    columns = 23
    arena: simulation.Arena = smart_agent.cvar.arena
    width = columns * block_size
    height = rows * block_size

    create_gui(arena)
    display_gui()
    while dpg.is_dearpygui_running():
        jobs = dpg.get_callback_queue()
        dpg.run_callbacks(jobs)

        update_text()
        
        dpg.render_dearpygui_frame()
    dpg.destroy_context()