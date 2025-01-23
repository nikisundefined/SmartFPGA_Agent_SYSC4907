import simulation
import dearpygui.dearpygui as dpg
import time

def create_texture(n: int, m: int) -> list[float]:
    texture_data: list[float] = []
    for _ in range(0, n * m):
        texture_data.append(255 / 255)
        texture_data.append(0)
        texture_data.append(255 / 255)
        texture_data.append(255 / 255)
    return texture_data

# Updates the grid text with the current state of the arena
def update_grid(arena: simulation.Arena, block_size: int = 10, tag: str | int = 'Environment'):
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
    global arena
    arena.move(simulation.Direction(user_data))
    if arena.on_goal():
        arena.set_goal()
        dpg.set_value('score', f'Score: {arena.player.score}')
    update_grid(arena)

def create_gui(arena: simulation.Arena, rows: int = 23, columns = 23, block_size: int = 10):
    VIEWPORT_WIDTH: int = round((rows * block_size) / 100.0 + 0.5) * 100
    VIEWPORT_HEIGHT: int = round((columns * block_size) / 100.0 + 0.5) * 100

    texture_width: int = rows * block_size
    texture_height: int = columns * block_size

    dpg.create_context()
    dpg.create_viewport(title='Pacman [float]', width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT, vsync=False)
    dpg.setup_dearpygui()

    arena_texture: list[float] = create_texture(texture_width, texture_height)
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width=texture_width, height=texture_height, default_value=arena_texture, tag='Environment')
    update_grid(arena)

    with dpg.window(tag="Pacman"):
        hori_offset = (dpg.get_viewport_width() - texture_width) / 2
        vert_offset = (dpg.get_viewport_height() - texture_height) / 2
        dpg.add_image("Environment", width=texture_width, height=texture_height, pos=[hori_offset, vert_offset])
        dpg.add_text(f'Score: {arena.player.score}', tag='score')
        dpg.add_text('Time: 0.0', tag='time')
    
    dpg.add_value_registry(tag='value_registry')
    dpg.add_float_value(tag='timer', parent='value_registry')

    with dpg.handler_registry():
        dpg.add_key_press_handler(key = dpg.mvKey_W, user_data = simulation.Direction.UP, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_A, user_data = simulation.Direction.LEFT, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_S, user_data = simulation.Direction.DOWN, callback=move)
        dpg.add_key_press_handler(key = dpg.mvKey_D, user_data = simulation.Direction.RIGHT, callback=move)

def display_gui():
    dpg.show_viewport()
    dpg.set_value('timer', time.time())
    dpg.set_primary_window("Pacman", True)

def update_text():
    start_time = dpg.get_value('timer')
    txt_rect = dpg.get_item_rect_size('score')
    dpg.set_item_pos('score', [dpg.get_viewport_width()/2-txt_rect[0]/2, 0])
    tim_rect = dpg.get_item_rect_size('time')
    dpg.set_item_pos('time', [dpg.get_viewport_width()/2-tim_rect[0]/2, txt_rect[1]])
    dpg.set_value('time', f'Time: {round(time.time() - start_time, 1)}')

if __name__ == "__main__":
    import time
    # Constants
    block_size = 10 # 10px * 10px
    rows = 23
    columns = 23
    arena: simulation.Arena = simulation.Arena()
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