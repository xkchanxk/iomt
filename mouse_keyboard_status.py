# mouse_keyboard_status.py

from pynput import mouse, keyboard
import threading
import time

_last_mouse_time = 0
_last_keyboard_time = 0
IDLE_TIMEOUT = 1.0  # 秒数内有活动则视为活跃

# 监听回调
def _on_mouse_activity(*args):
    global _last_mouse_time
    _last_mouse_time = time.time()

def _on_keyboard_activity(*args):
    global _last_keyboard_time
    _last_keyboard_time = time.time()

# 启动监听器（只启动一次）
_started = False
def _start_listeners_once():
    global _started
    if _started:
        return
    _started = True

    mouse.Listener(on_move=_on_mouse_activity,
                   on_click=_on_mouse_activity,
                   on_scroll=_on_mouse_activity).start()
    keyboard.Listener(on_press=_on_keyboard_activity).start()

# 对外接口函数
def get_mouse_keyboard_status():
    _start_listeners_once()
    now = time.time()
    mouse_recent = (now - _last_mouse_time) < IDLE_TIMEOUT
    keyboard_recent = (now - _last_keyboard_time) < IDLE_TIMEOUT
    return mouse_recent, keyboard_recent