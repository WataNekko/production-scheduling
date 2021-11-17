import numpy as np
from pygame import surface
from pygame.mouse import get_pos


def read_excel(path: str, cells: str, sheet_name: str = None) -> np.ndarray:
    """Read an Excel file and return an array of cell values."""

    from openpyxl import load_workbook

    # open excel file
    wb = load_workbook(filename=path)
    ws = wb.active if sheet_name is None else wb[sheet_name]

    # read cells
    cells_tuple = ws[cells]
    get_values = np.vectorize(lambda cell: cell.value, otypes=[object])

    return get_values(cells_tuple)


def visualize(process_time, start_time) -> None:
    """
    Visualize the scheduling of a production line problem.

    Parameters:
        process_time: A table representing the processing time of the jobs on different machines,
            where the rows are the jobs and columns are the machines.
        start_time: A table representing the time to start processing the jobs on different machines,
            where the rows are the jobs and columns are the machines.
    """

    import pygame
    import random
    import time

    # input validation
    process_time = np.array(process_time)
    non_none_process_time = process_time[process_time != None]

    start_time = np.array(start_time)
    non_none_start_time = start_time[start_time != None]

    if process_time.shape != start_time.shape or np.any(
        (non_none_process_time < 0) | (non_none_start_time < 0)
    ):
        raise ValueError(
            "The two input tables should be of the same dimensions and all values must be non-negative or None."
        )

    # pygame initialization
    pygame.init()
    pygame.display.set_caption("Scheduling of a production line")

    resolution = (1000, 600)
    screen = pygame.display.set_mode(resolution)

    black = (0, 0, 0)
    white = (255, 255, 255)
    font = pygame.font.SysFont("Courier New", 50, bold=True)
    mach_cmpn_size = 124
    job_cmpn_size = 110

    # components
    class Component:
        def __init__(
            self,
            surface: pygame.Surface,
            pos: tuple[int, int] = None,
            children: list["Component"] = None,
        ):
            self._surface = surface
            self._pos = pos if pos is not None else (0, 0)
            self._children = children if children is not None else []

            self._parent: Component = None
            for child in self._children:
                child._parent = self
            self._movement = None

            self.set_dirty()

        def set_dirty(self):
            """Set the dirty flag for this component. Dirty components are repainted every rendering update."""
            self._is_dirty = True

        def render(self, surface: pygame.Surface):
            surface.blit(self._surface, self._pos)

        def repaint(self):
            """Override this method in child classes to implement the components' look.
            There must be a call to super().repaint() at the end of the method."""

            # clear first if it's an empty Component
            if type(self) is Component:
                self._surface.fill((0, 0, 0, 0))

            for child in self._children:
                child.render(self._surface)

        def render_update(self):
            """Call this method on the root component every frame to process rendering updates."""

            # recursively checking the chain from bottom up
            for child in self._children:
                child.render_update()

            if self._is_dirty:
                self.repaint()
                self._is_dirty = False
                if self._parent is not None:
                    # setting dirty up the parent to repaint the affected chain
                    self._parent.set_dirty()
                else:
                    # if we just repainted the root component (parent is None),
                    # then render it to the screen.
                    screen.fill(black)
                    self.render(screen)

        def add_child(self, component: "Component"):
            self._children.append(component)
            component._parent = self
            self.set_dirty()

        def update(self):
            """Override this method in child classes to implement components' update logic.
            There must a call to super().update() at the beginning of the method."""

            for child in self._children:
                child.update()

            if self._movement is not None:
                target: np.ndarray
                curr_pos: np.ndarray
                duration: float
                prev_time: float
                target, [curr_pos, duration, prev_time] = self._movement

                curr_time = time.time()
                delta_time = curr_time - prev_time
                progress = delta_time / duration

                if progress >= 1:
                    # passed the movement duration
                    self._pos = target
                    self._movement = None
                else:
                    # update pos
                    curr_pos = curr_pos + (target - curr_pos) * progress
                    self._pos = curr_pos

                    # update the movement object
                    self._movement[1][0] = curr_pos
                    self._movement[1][1] = duration - delta_time
                    self._movement[1][2] = curr_time

                self.set_dirty()

        def move_pos(self, target: tuple[int, int], duration: float = 0.0):
            if duration > 0:
                # record a movement object to be processed in update()
                self._movement = (
                    np.array(target),
                    [np.array(self._pos), duration, time.time()],
                )
            else:
                self._pos = target
                self._movement = None
                self.set_dirty()

        def get_size(self):
            return self._surface.get_size()

        def get_pos_center(
            self, target: pygame.Surface | tuple[int, int]
        ) -> tuple[int, int]:
            x, y = self.get_size()
            if isinstance(target, pygame.Surface):
                width, height = target.get_size()
                return round((width - x) / 2), round((height - y) / 2)
            else:
                return round(target[0] - x / 2), round(target[1] - y / 2)

    class Label(Component):
        def __init__(
            self,
            text: str = "Label",
            color: tuple[int, int, int] = white,
            font: pygame.font.Font = font,
            pos: tuple[int, int] = None,
            children: list[Component] = None,
        ):
            self._text = text
            self._color = color
            self._font = font
            super().__init__(None, pos, children)

            # pre-paint before the rendering update frame to ensure the surface is
            # available after this constructor.
            self.repaint()
            self._is_dirty = False

        def repaint(self):
            self._surface = self._font.render(self._text, True, self._color)
            super().repaint()

        def set_color(self, color: tuple[int, int, int]):
            self._color = color
            self.set_dirty()

    class LabeledComponent(Component):
        def __init__(
            self,
            label: str,
            fgcolor: tuple[int, int, int],
            font: pygame.font.Font,
            bgcolor: tuple[int, int, int],
            border_color: tuple[int, int, int],
            size: tuple[int, int],
            pos: tuple[int, int],
            children: list[Component],
            is_background_alpha: bool = False,
        ):
            flags = pygame.SRCALPHA if is_background_alpha else 0
            surface = pygame.Surface(size, flags)

            lbl = Label(label, fgcolor, font)
            lbl._pos = lbl.get_pos_center(surface)
            self._label = lbl

            self._bgcolor = bgcolor
            self._border_color = border_color

            super().__init__(surface, pos, children)

        def repaint(self):
            self._label.render(self._surface)
            super().repaint()

    class Machine(LabeledComponent):
        def __init__(
            self,
            label: str = "M",
            fgcolor: tuple[int, int, int] = white,
            font: pygame.font.Font = font,
            bgcolor: tuple[int, int, int] = (128, 25, 59),
            border_color: tuple[int, int, int] = (80, 0, 47),
            size: tuple[int, int] = (mach_cmpn_size, mach_cmpn_size),
            pos: tuple[int, int] = None,
            children: list[Component] = None,
        ):
            super().__init__(
                label, fgcolor, font, bgcolor, border_color, size, pos, children
            )

        def repaint(self):
            self._surface.fill(self._bgcolor)
            if self._border_color is not None:
                pygame.draw.rect(
                    self._surface, self._border_color, self._surface.get_rect(), width=9
                )
            super().repaint()

    class Job(LabeledComponent):
        def __init__(
            self,
            label: str = "J",
            fgcolor: tuple[int, int, int] = white,
            font: pygame.font.Font = font,
            bgcolor: tuple[int, int, int] = (0, 158, 26),
            border_color: tuple[int, int, int] = (0, 75, 12),
            size: tuple[int, int] = (job_cmpn_size, job_cmpn_size),
            pos: tuple[int, int] = None,
            children: list[Component] = None,
        ):
            super().__init__(
                label,
                fgcolor,
                font,
                bgcolor,
                border_color,
                size,
                pos,
                children,
                is_background_alpha=True,
            )

        def repaint(self):
            self._surface.fill((0, 0, 0, 0))  # clear the surface
            rect = self._surface.get_rect()
            pygame.draw.ellipse(self._surface, self._bgcolor, rect)
            if self._border_color is not None:
                pygame.draw.ellipse(self._surface, self._border_color, rect, width=6)
            super().repaint()

    class PlayButton:
        pass

    class Slider:
        pass

    class ProgressBar:
        pass

    class SchedulingChart:
        pass

    class RestartButton:
        pass

    # random color
    random.seed(5)

    def get_color():
        return tuple(random.randint(0, 255) for _ in range(3))

    # component list
    num_jobs, num_machines = process_time.shape

    def new_job(i):
        """Helper function to generate Job with random color"""
        bg = get_color()
        border = [darkened if (darkened := c - 50) >= 0 else 0 for c in bg]
        return Job(f"J{i + 1}", bgcolor=bg, border_color=border)

    jobs = [new_job(i) for i in range(num_jobs)]
    machines = [Machine(f"M{i + 1}") for i in range(num_machines)]

    components: list[Component] = machines + jobs
    content = Component(
        pygame.Surface(resolution, pygame.SRCALPHA), children=components
    )

    # position the components
    width, _ = resolution
    preferred_margin, margin_y = 40, 50
    padding_x = 50

    get_num_per_row = lambda size, preferred_margin: int(
        (width - 2 * padding_x - preferred_margin) / (size + preferred_margin)
    )

    get_real_margin = lambda size, num: (width - 2 * padding_x - size * num) / (num + 1)

    get_start_x = lambda size, num, margin: (
        (width - size * margin - margin * (num - 1)) / 2
    )

    mach_per_row = get_num_per_row(mach_cmpn_size, preferred_margin)
    job_per_row = get_num_per_row(job_cmpn_size, preferred_margin)

    mach_margin = get_real_margin(mach_cmpn_size, mach_per_row)
    job_margin = get_real_margin(job_cmpn_size, job_per_row)

    def get_pos_array(
        total_num, num_per_row, start_x, start_y, size, margin_x, margin_y
    ) -> np.ndarray:
        row = [(start_x + (size + margin_x) * i, start_y) for i in range(num_per_row)]
        num_whole_rows = int(total_num / num_per_row)
        num_rem = total_num % num_per_row

        arr = np.empty(shape=(0, 2))
        for i in range(num_whole_rows):
            new_row = [(x, y + (size + margin_y) * i) for x, y in row]
            arr = np.append(arr, new_row, 0)
        if num_rem > 0:
            x = get_start_x(size, num_rem, margin_x)
            y = start_y + (size + margin_y) * num_whole_rows
            row = [(x + (size + margin_x) * i, y) for i in range(num_rem)]
            arr = np.append(arr, row, 0)

        return arr

    print(
        get_pos_array(
            num_machines,
            mach_per_row,
            get_start_x(mach_cmpn_size, mach_per_row, mach_margin),
            50,
            mach_cmpn_size,
            mach_margin,
            margin_y,
        )
    )

    try:
        # main loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for c in components:
                        x = random.randint(0, resolution[0])
                        y = random.randint(0, resolution[1])
                        c.move_pos((x, y), 0.3)

            # update components
            content.update()

            # render display
            content.render_update()
            pygame.display.update()

    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    p = read_excel("test.xlsx", "b2:e4")
    s = read_excel("test.xlsx", "b7:e9")
    visualize(p, s)
