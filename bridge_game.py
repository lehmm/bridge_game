import pygame
import pymunk
import math

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bridge Builder - Steps 12–16")

clock = pygame.time.Clock()

# --- Build-mode data ---
nodes = []              # [(x, y), ...]
node_is_anchor = []     # [bool, ...]
beams = []              # [(node_index_a, node_index_b), ...]

selected_node = None
hovered_node = None
simulate_mode = False

# --- Undo / Redo history ---
history = []            # List of (nodes, node_is_anchor, beams)
redo_stack = []         # List of undone states to redo

def push_history():
    """Save a snapshot of the current build state for undo/redo."""
    global history, redo_stack
    history.append((
        nodes.copy(),
        node_is_anchor.copy(),
        beams.copy()
    ))
    redo_stack.clear()  # Clear redo history when a new change is made

def undo():
    """Revert to the previous build state."""
    global nodes, node_is_anchor, beams, history, redo_stack
    if len(history) > 1:
        state = history.pop()          # Pop current
        redo_stack.append(state)       # Save it to redo
        prev_nodes, prev_anchors, prev_beams = history[-1]
        nodes = prev_nodes.copy()
        node_is_anchor = prev_anchors.copy()
        beams = prev_beams.copy()

def redo():
    """Re-apply the last undone build state."""
    global nodes, node_is_anchor, beams, history, redo_stack
    if redo_stack:
        state = redo_stack.pop()
        history.append(state)
        nodes, node_is_anchor, beams = (
            state[0].copy(),
            state[1].copy(),
            state[2].copy()
        )

def clear_all():
    global nodes, node_is_anchor, beams
    nodes = []
    node_is_anchor = []
    beams = []

def draw_legend(screen):
    """Draw a semi-transparent legend box that automatically sizes to text."""
    font = pygame.font.SysFont("consolas", 16)

    lines = [
        "BUILD CONTROLS:",
        "Left Click  - Place / Connect Node",
        "Right Click - Toggle Anchor",
        "Middle Click- Delete Node / Beam",
        "Z - Undo",
        "Y - Redo",
        "C - Clear All",
        "SPACE - Start Simulation"
    ]

    padding = 8
    line_height = 18

    # Measure the widest text line
    max_width = 0
    for txt in lines:
        w, h = font.size(txt)
        if w > max_width:
            max_width = w

    # Final box size
    width = max_width + padding * 2
    height = padding * 2 + line_height * len(lines)

    # Semi-transparent box
    legend_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    legend_surface.fill((0, 0, 0, 160))

    # Draw each line of text
    for i, txt in enumerate(lines):
        label = font.render(txt, True, (255, 255, 255))
        legend_surface.blit(label, (padding, padding + i * line_height))

    # Draw to screen
    screen.blit(legend_surface, (10, 10))



# ------------- Build-mode selection helpers -----------------

def find_node_at_position(x, y):
    """Return index of a node near the cursor, or None."""
    pick_radius = 8
    r2 = pick_radius * pick_radius
    for i, (nx, ny) in enumerate(nodes):
        if (x - nx)**2 + (y - ny)**2 < r2:
            return i
    return None

def find_beam_at_position(x, y):
    """Return index of a beam under cursor by distance to segment."""
    px, py = x, y
    threshold = 6.0
    best_index = None
    best_dist = float("inf")

    for i, (a, b) in enumerate(beams):
        ax, ay = nodes[a]
        bx, by = nodes[b]

        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay

        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0:
            continue

        t = (wx * vx + wy * vy) / seg_len2
        t = max(0.0, min(1.0, t))

        cx = ax + vx * t
        cy = ay + vy * t

        dist = math.hypot(px - cx, py - cy)
        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_index = i

    return best_index

def delete_node(index):
    """Delete node + all attached beams, reindex beams above it."""
    global nodes, node_is_anchor, beams

    del nodes[index]
    del node_is_anchor[index]

    new_beams = []
    for a, b in beams:
        if a == index or b == index:
            continue  # beam touching removed node
        if a > index: a -= 1
        if b > index: b -= 1
        new_beams.append((a, b))
    beams = new_beams


# ------------- Physics structures -----------------

space = None
pm_nodes = []
pm_joints_pin = []
pm_joints_spring = []
rest_lengths = []

pm_ground = None
pm_load = None

beam_indices = []
pm_beam_bodies = []
pm_beam_shapes = []

static_beams = []
pm_beam_bodies_static = []
pm_beam_shapes_static = []

pm_bend_springs = []
beam_stress = {}   # {frozenset((a,b)): strain}


# ------------- Physics setup helpers -----------------

def create_ground(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (0, HEIGHT - 10), (WIDTH, HEIGHT - 10), 5)
    shape.friction = 1.0
    space.add(body, shape)
    return shape

def create_load(space):
    mass = 5
    size = 20
    moment = pymunk.moment_for_box(mass, (size, size))
    body = pymunk.Body(mass, moment)
    body.position = (WIDTH // 2, 50)
    shape = pymunk.Poly.create_box(body, (size, size))
    shape.friction = 0.8
    space.add(body, shape)
    return body

def create_beam_collider(space, A_pos, B_pos):
    """Create a kinematic collision segment that matches the beam."""
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    ax, ay = A_pos
    bx, by = B_pos
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        length = 1.0

    mid_x = (ax + bx) / 2
    mid_y = (ay + by) / 2
    body.position = (mid_x, mid_y)
    body.angle = math.atan2(dy, dx)

    half = length / 2
    shape = pymunk.Segment(body, (-half, 0), (half, 0), 2)
    shape.friction = 0.9

    space.add(body, shape)
    return body, shape


# ------------- Start Simulation -----------------

def start_simulation():
    global space, pm_nodes, pm_joints_pin, pm_joints_spring, rest_lengths
    global pm_ground, pm_load
    global beam_indices, pm_beam_bodies, pm_beam_shapes
    global static_beams, pm_beam_bodies_static, pm_beam_shapes_static
    global pm_bend_springs, beam_stress

    space = pymunk.Space()
    space.gravity = (0, 900)

    pm_nodes = []
    pm_joints_pin = []
    pm_joints_spring = []
    rest_lengths = []

    beam_indices = []
    pm_beam_bodies = []
    pm_beam_shapes = []

    static_beams = []
    pm_beam_bodies_static = []
    pm_beam_shapes_static = []

    pm_bend_springs = []
    beam_stress = {}

    pm_ground = create_ground(space)

    # Node bodies
    for (x, y), anchor in zip(nodes, node_is_anchor):
        if anchor:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        body.position = (x, y)
        shape = pymunk.Circle(body, 5)
        shape.friction = 1.0
        space.add(body, shape)
        pm_nodes.append(body)

    stiffness = 1000
    damping = 25
    angle_stiffness = stiffness * 0.3
    angle_damping = damping * 0.5

    # Create beams + springs
    for a, b in beams:
        A = pm_nodes[a]
        B = pm_nodes[b]

        if node_is_anchor[a] and node_is_anchor[b]:
            # static beam
            body, shape = create_beam_collider(space, A.position, B.position)
            static_beams.append((a, b))
            pm_beam_bodies_static.append(body)
            pm_beam_shapes_static.append(shape)
            beam_stress[frozenset((a, b))] = 0.0
            continue

        # dynamic beam
        rest = (A.position - B.position).length
        rest_lengths.append(rest)

        pin = pymunk.PinJoint(A, B)
        spring = pymunk.DampedSpring(A, B, (0, 0), (0, 0), rest, stiffness, damping)
        space.add(pin, spring)

        pm_joints_pin.append(pin)
        pm_joints_spring.append(spring)
        beam_indices.append((a, b))

        body, shape = create_beam_collider(space, A.position, B.position)
        pm_beam_bodies.append(body)
        pm_beam_shapes.append(shape)

        beam_stress[frozenset((a, b))] = 0.0

    # Add soft bending springs
    adjacency = {}
    for a, b in beam_indices:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    for center, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1 = neighbors[i]
                n2 = neighbors[j]
                A = pm_nodes[n1]
                B = pm_nodes[n2]
                rest = (A.position - B.position).length
                spring = pymunk.DampedSpring(
                    A, B, (0, 0), (0, 0), rest, angle_stiffness, angle_damping
                )
                space.add(spring)
                pm_bend_springs.append(spring)

    pm_load = create_load(space)


def stop_simulation():
    global space, pm_nodes, pm_joints_pin, pm_joints_spring, rest_lengths
    global pm_ground, pm_load
    global beam_indices, pm_beam_bodies, pm_beam_shapes
    global static_beams, pm_beam_bodies_static, pm_beam_shapes_static
    global pm_bend_springs, beam_stress

    space = None
    pm_nodes = []
    pm_joints_pin = []
    pm_joints_spring = []
    rest_lengths = []
    pm_load = None

    beam_indices = []
    pm_beam_bodies = []
    pm_beam_shapes = []

    static_beams = []
    pm_beam_bodies_static = []
    pm_beam_shapes_static = []

    pm_bend_springs = []
    beam_stress = {}


# ------------- Physics step helpers -----------------

def update_beam_collision_shapes():
    """Update kinematic beam colliders to match current node positions."""
    # Dynamic beams
    for (a, b), body, shape in zip(beam_indices, pm_beam_bodies, pm_beam_shapes):
        A = pm_nodes[a].position
        B = pm_nodes[b].position
        dx = B.x - A.x
        dy = B.y - A.y
        length = math.hypot(dx, dy)

        mid = (A + B) / 2
        body.position = mid
        body.angle = math.atan2(dy, dx)

        half = length / 2
        shape.unsafe_set_endpoints((-half, 0), (half, 0))

    # Static beams
    for (a, b), body, shape in zip(static_beams, pm_beam_bodies_static, pm_beam_shapes_static):
        A = pm_nodes[a].position
        B = pm_nodes[b].position
        dx = B.x - A.x
        dy = B.y - A.y
        length = math.hypot(dx, dy)

        mid = (A + B) / 2
        body.position = mid
        body.angle = math.atan2(dy, dx)

        half = length / 2
        shape.unsafe_set_endpoints((-half, 0), (half, 0))


def stress_to_color(stress):
    """Map strain value to RGB: tension=red, compression=blue."""
    limit = 0.5
    stress = max(-limit, min(limit, stress))

    if stress >= 0:
        t = stress / limit
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    else:
        t = (-stress) / limit
        r = int(255 * (1 - t))
        g = int(255 * (1 - t))
        b = 255

    return (r, g, b)


# ------------------ Initialize empty history ---------------
push_history()


# --------------------------- Main loop ----------------------

running = True
while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    hovered_node = find_node_at_position(mouse_x, mouse_y)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Toggle simulation mode
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            simulate_mode = not simulate_mode
            if simulate_mode:
                start_simulation()
            else:
                stop_simulation()

        # Build-mode only: Undo / Redo / Clear
        if not simulate_mode and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z:
                undo()
            elif event.key == pygame.K_y:
                redo()
            elif event.key == pygame.K_c:
                clear_all()
                push_history()

        # Build mode interaction
        if not simulate_mode:

            # Left click: create node or connect nodes
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = find_node_at_position(mouse_x, mouse_y)

                if clicked is None:
                    nodes.append((mouse_x, mouse_y))
                    node_is_anchor.append(False)
                    push_history()
                else:
                    if selected_node is None:
                        selected_node = clicked
                    else:
                        a = selected_node
                        b = clicked
                        if a != b and (a, b) not in beams and (b, a) not in beams:
                            beams.append((a, b))
                            push_history()
                        selected_node = None

            # Right click: toggle anchor
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                clicked = find_node_at_position(mouse_x, mouse_y)
                if clicked is not None:
                    node_is_anchor[clicked] = not node_is_anchor[clicked]
                    push_history()

            # Middle click: delete node or beam
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                clicked_node = find_node_at_position(mouse_x, mouse_y)
                if clicked_node is not None:
                    delete_node(clicked_node)
                    push_history()
                else:
                    beam_index = find_beam_at_position(mouse_x, mouse_y)
                    if beam_index is not None:
                        del beams[beam_index]
                        push_history()

    # ------------------ PHYSICS UPDATE ---------------------

    if simulate_mode and space is not None:

        update_beam_collision_shapes()
        space.step(1 / 60)

        # Beam breaking and stress updates
        break_ratio = 1.4
        plastic_ratio = 1.05
        compress_ratio = 0.7

        new_beam_indices = []
        new_rest = []
        new_pins = []
        new_springs = []
        new_bodies = []
        new_shapes = []

        for (a, b), rest, pin, spring, body, shape in zip(
            beam_indices, rest_lengths,
            pm_joints_pin, pm_joints_spring,
            pm_beam_bodies, pm_beam_shapes
        ):
            A = pm_nodes[a].position
            B = pm_nodes[b].position
            dist = (A - B).length
            key = frozenset((a, b))

            # Break under tension or compression
            if dist > rest * break_ratio or dist < rest * compress_ratio:
                space.remove(pin, spring)
                space.remove(body, shape)
                if (a, b) in beams:
                    beams.remove((a, b))
                elif (b, a) in beams:
                    beams.remove((b, a))
                beam_stress.pop(key, None)
                continue

            # Plastic deformation
            if dist > rest * plastic_ratio:
                rest = dist
                spring.rest_length = dist

            # Stress = strain
            if rest != 0:
                strain = (dist - rest) / rest
            else:
                strain = 0
            beam_stress[key] = strain

            new_beam_indices.append((a, b))
            new_rest.append(rest)
            new_pins.append(pin)
            new_springs.append(spring)
            new_bodies.append(body)
            new_shapes.append(shape)

        beam_indices = new_beam_indices
        rest_lengths = new_rest
        pm_joints_pin = new_pins
        pm_joints_spring = new_springs
        pm_beam_bodies = new_bodies
        pm_beam_shapes = new_shapes

    # --------------------- DRAWING -------------------------

    screen.fill((0, 0, 0))

    # Draw beams
    if simulate_mode:
        for (a, b) in beams:
            ax, ay = pm_nodes[a].position
            bx, by = pm_nodes[b].position
            color = stress_to_color(beam_stress.get(frozenset((a, b)), 0.0))
            pygame.draw.line(screen, color, (ax, ay), (bx, by), 2)
    else:
        for a, b in beams:
            ax, ay = nodes[a]
            bx, by = nodes[b]
            pygame.draw.line(screen, (255, 255, 255), (ax, ay), (bx, by), 2)

    # Draw nodes
    if simulate_mode:
        for body, anchor in zip(pm_nodes, node_is_anchor):
            x, y = body.position
            color = (0, 180, 255) if anchor else (255, 255, 255)
            pygame.draw.circle(screen, color, (int(x), int(y)), 5)
    else:
        for i, ((x, y), anchor) in enumerate(zip(nodes, node_is_anchor)):
            if i == hovered_node:
                pygame.draw.circle(screen, (100, 100, 100), (x, y), 7)
            if i == selected_node:
                pygame.draw.circle(screen, (150, 150, 150), (x, y), 7)
            color = (0, 180, 255) if anchor else (255, 255, 255)
            pygame.draw.circle(screen, color, (x, y), 5)

    # Draw load
    if simulate_mode and pm_load is not None:
        x, y = pm_load.position
        pygame.draw.rect(screen, (255, 0, 0), (int(x) - 10, int(y) - 10, 20, 20))

    # Draw legend only in build mode
    if not simulate_mode:
        draw_legend(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
