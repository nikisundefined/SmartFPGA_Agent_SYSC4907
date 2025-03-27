from . import vars
from . import shared
from . import simulation

import logging
import sys
import threading
import time

### Global initialization
log: logging.Logger = logging.getLogger('smart_agent')
handle = logging.StreamHandler(sys.stdout)
handle.setFormatter(logging.Formatter(f'{threading.current_thread().name}[{{name}}] | {{levelname}} -> {{message}}', style='{'))
log.addHandler(handle)

start_time: float = time.time()

if 'svar' not in globals():
    svar: vars.SharedDict = vars.SharedDict()
    log.debug('Initialized shared variables')

# If console vars are not loaded, attempt to load them from shared memory
if 'cvar' not in globals():
    log.debug('cvar not found, attempting to load from shared memory')
    try:
        tmp: memoryview = shared.create_shared_memory(shared.SharedConsoleDict.size, svar.cvars_name, shared.AttachFlag.ATTACH)
        log.debug(f'Loaded Console Variables from shared memory with name: {svar.cvars_name}')
    except Exception as e:
        log.warning('Could not load Console Variables from shared memory, creating new instance')
        tmp: memoryview = shared.create_shared_memory(shared.SharedConsoleDict.size, svar.cvars_name)
    cvar: vars.DefaultConsoleDict = shared.SharedConsoleDict(tmp)
    log.setLevel(cvar.log_level)

# If GUI vars are not loaded, attempt to load them from shared memory
if 'gvar' not in globals():
    log.debug('gvar not found, attempting to load from shared memory')
    try:
        tmp: memoryview = shared.create_shared_memory(shared.SharedGUIDict.size, svar.gvars_name, shared.AttachFlag.ATTACH)
        log.debug(f'Loaded GUI Variables from shared memory with name: {svar.gvars_name}')
    except:
        log.warning('Could not load GUI Variables from shared memory, creating new instance')
        tmp: memoryview = shared.create_shared_memory(shared.SharedGUIDict.size, svar.gvars_name)
    gvar: vars.GUIDict = shared.SharedGUIDict(tmp)
    gvar.arena = cvar.arena

end_time: float = time.time()
log.debug(f'smart_agent load complete, took: {round(end_time-start_time,2)}s')