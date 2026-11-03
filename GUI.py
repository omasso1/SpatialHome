from ursina import Slider, Vec2, Vec3, camera, window, Button
from typing import Callable

class GUI:
    class Slider_config:
        def __init__(self, min: int,
                     max: int,
                     default: int,
                     position: Vec2 = Vec2(0, 0),
                     step: int = 1,
                     callback: Callable | None = None,
                     dynamic: bool = False):

            self.min = min
            self.max = max
            self.default = default
            self.step = step
            self.position = position
            self.callback = callback
            self.dynamic = dynamic

    class Button_config:
            def __init__(self,
                         text: str,
                         position: Vec2 = Vec2(0, 0),
                         callback: Callable | None = None,
                         ):
                self.text = text
                self.position = position
                self.callback = callback

    def __init__(self):
        self.sliders: dict[str, Slider] = {}
        self.buttons: dict[str, Button] = {}

    def add_slider(self, slider_name: str, slider_config: Slider_config) -> None:
        new_slider = Slider(
            slider_config.min,
            slider_config.max,
            slider_config.default
        )
        new_slider.parent = camera.ui
        new_slider.step = slider_config.step
        new_slider.position = slider_config.position + window.top_left
        new_slider.on_value_changed = slider_config.callback
        new_slider.dynamic = slider_config.dynamic
        self.sliders[slider_name] = new_slider

    def add_button(self, button_name: str, button_config: Button_config) -> None:
        new_button = Button(text=button_config.text)
        new_button.on_click = button_config.callback
        new_button.fit_to_text()
        new_button.position = button_config.position + window.top_left
        self.buttons[button_name] = new_button

