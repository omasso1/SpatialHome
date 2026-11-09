import time

delta_time: int = 0
previous_update_time: int = 0

def update():
    global previous_update_time
    global delta_time

    t = int(time.time() * 1000)
    delta_time = t - previous_update_time
    previous_update_time = int(t)