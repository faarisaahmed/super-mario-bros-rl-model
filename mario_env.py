import gym
from gym import spaces
import numpy as np
import pygame
import math

from mario import Mario
from level_loader import Level
from sprite_manager import SpriteManager

SCALE = 3
BASE_WIDTH, BASE_HEIGHT = 256, 240
SCREEN_WIDTH = BASE_WIDTH * SCALE
SCREEN_HEIGHT = BASE_HEIGHT * SCALE

# Matches mario.py exactly: self.gravity = 800 * scale
GRAVITY_PER_SECOND = 800.0 * SCALE


class MarioEnv(gym.Env):
    def __init__(self, render_mode=False):
        pygame.init()
        pygame.display.set_mode((1, 1))
        super().__init__()

        self.render_mode = render_mode
        self.level = Level("levels/1-1.json", "tileset.json")
        self.mario = Mario(x=100, y=0, scale=SCALE)
        self.sprites = SpriteManager("sprites", SCALE)

        self.camera_x = 0
        self.prev_x = self.mario.x
        self.steps_since_progress = 0   # detects the agent getting stuck

        # Jump hold state — env simulates button hold internally
        self._jump_frames_left = 0
        self._jump_holding = False

        # Actions:
        # 0: idle
        # 1: walk right
        # 2: walk left
        # 3: short hop + right  (~6 frames hold → low arc, short gap)
        # 4: full jump + right  (~18 frames hold → max arc, wide gap)
        # 5: full jump only     (no horizontal, for jumping over walls)
        self.action_space = spaces.Discrete(6)

        # Observation (18 values):
        #  0-1:   mario x, y
        #  2-3:   velocity_x, velocity_y
        #  4:     on_ground (bool)
        #  5-9:   ground sensors at 16, 32, 64, 96, 128 px ahead
        #  10-14: wall sensors at 16, 32, 48, 64, 80 px ahead (torso height)
        #  15:    estimated landing_x for full jump
        #  16:    gap_start_x (0.0 if no gap found)
        #  17:    gap_end_x   (0.0 if no gap found)
        self.observation_space = spaces.Box(
            low=-100_000, high=100_000,
            shape=(18,), dtype=np.float32
        )

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def reset(self):
        self.level = Level("levels/1-1.json", "tileset.json")
        self.mario = Mario(x=100, y=0, scale=SCALE)
        self.camera_x = 0
        self.prev_x = self.mario.x
        self.steps_since_progress = 0
        self._jump_frames_left = 0
        self._jump_holding = False
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action):
        dt = 1 / 60
        done = False
        reward = 0.0

        was_on_ground = self.mario.on_ground
        x_before = self.mario.x

        self._apply_action(action, was_on_ground)

        solid_tiles = self.level.get_solid_tiles(SCALE)
        self.mario.update(dt, self.camera_x, solid_tiles)
        self._update_camera()

        delta_x = self.mario.x - self.prev_x

        # ── 1. Progress reward ────────────────────────────────────────
        # Full credit on ground, reduced in air so jumping isn't free
        if self.mario.on_ground:
            reward += np.clip(delta_x, -8, 8) * 0.1
        else:
            reward += np.clip(delta_x, -8, 8) * 0.03

        # Penalise moving backwards
        if delta_x < 0:
            reward -= 0.05

        # ── 2. Stuck detection ────────────────────────────────────────
        # If Mario hasn't made meaningful forward progress for 180 frames
        # (3 seconds), apply a growing penalty to unstick him
        if delta_x > 2.0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        if self.steps_since_progress > 180:
            reward -= 0.05  # escalating pressure to move

        # ── 3. Grounded bonus / air tax ───────────────────────────────
        if self.mario.on_ground:
            reward += 0.02
        else:
            reward -= 0.01

        # ── 4. Wall collision detection ───────────────────────────────
        # If Mario tried to move right but barely moved, he hit a wall
        if action in [1, 3, 4] and abs(delta_x) < 1.0 and was_on_ground:
            reward -= 0.08

        # ── 5. Jump evaluation (fires once, on the frame of the jump) ─
        jumped_this_step = action in [3, 4, 5] and was_on_ground
        if jumped_this_step:
            gap_start, gap_end = self._find_gap_ahead(max_look=250)
            wall_close = self._wall_ahead(24)
            landing_x  = self._estimate_landing_x(action)

            has_gap  = gap_start is not None
            has_wall = wall_close

            if has_gap and not has_wall:
                clears = (landing_x is not None) and (landing_x >= gap_end)
                if clears:
                    # Scale bonus with gap width — harder gaps earn more
                    gap_width = gap_end - gap_start
                    reward += 0.4 + min(gap_width / 300.0, 0.3)
                else:
                    reward -= 0.3   # jumped but won't clear the gap

            elif has_wall and not has_gap:
                # Jumping over a wall is fine; jumping INTO one is not
                if action == 5:
                    reward += 0.1   # deliberate wall-jump, good
                else:
                    reward -= 0.3   # ran and jumped into a wall

            elif not has_gap and not has_wall:
                reward -= 0.2       # unnecessary jump on flat ground

            # has_gap and has_wall together: ambiguous, small penalty
            elif has_gap and has_wall:
                reward -= 0.05

        # ── 6. Air spam penalty ───────────────────────────────────────
        if action in [3, 4, 5] and not was_on_ground:
            reward -= 0.1

        # ── 7. Death ──────────────────────────────────────────────────
        # Scaled down from -150 so value function can converge
        floor_y = self.level.height * self.level.tile_size * SCALE
        if self.mario.y >= floor_y:
            reward -= 15
            done = True

        # ── 8. Level completion ───────────────────────────────────────
        # Scaled down from +300 for the same reason
        level_pixel_width = self.level.width * self.level.tile_size * SCALE
        if self.mario.x >= level_pixel_width - 50:
            reward += 30
            done = True

        # Tiny living penalty — prevents idle loops
        reward -= 0.003

        self.prev_x = self.mario.x
        return self._get_state(), reward, done, {}

    # ------------------------------------------------------------------
    def _apply_action(self, action, was_on_ground):
        self.mario.velocity_x = 0

        # Tick down active jump hold
        if self._jump_frames_left > 0:
            self._jump_frames_left -= 1
            self._jump_holding = self._jump_frames_left > 0
        else:
            self._jump_holding = False

        if action == 0:
            pass  # idle

        elif action == 1:
            self.mario.velocity_x = self.mario.horizontal_speed

        elif action == 2:
            self.mario.velocity_x = -self.mario.horizontal_speed

        elif action == 3:
            # Short hop + right — low arc, clears small gaps
            self.mario.velocity_x = self.mario.horizontal_speed
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 6
                self._jump_holding = True

        elif action == 4:
            # Full jump + right — max arc, clears wide gaps
            self.mario.velocity_x = self.mario.horizontal_speed
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 18
                self._jump_holding = True

        elif action == 5:
            # Full jump, no horizontal — for wall situations
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 18
                self._jump_holding = True

        # While mid-hold and airborne, keep horizontal velocity going
        if self._jump_holding and not was_on_ground:
            if action in [3, 4]:
                self.mario.velocity_x = self.mario.horizontal_speed

        # Simulate held/released jump button via min_jump_velocity cut
        # mario.py applies this cut when Z is not held
        if self._jump_holding and self.mario.velocity_y < 0:
            pass  # holding — let the arc run fully
        elif not self._jump_holding and self.mario.velocity_y < self.mario.min_jump_velocity:
            self.mario.velocity_y = self.mario.min_jump_velocity  # cut arc short

    # ------------------------------------------------------------------
    def _ground_ahead(self, pixels_ahead):
        """True if a solid tile exists at foot level, pixels_ahead in front."""
        check_x = self.mario.x + pixels_ahead
        check_y = self.mario.y + (self.mario.height * self.mario.scale) + 5
        col = int(check_x // (self.level.tile_size * SCALE))
        row = int(check_y // (self.level.tile_size * SCALE))
        if not (0 <= row < self.level.height and 0 <= col < self.level.width):
            return False
        tile_id = self.level.grid[row][col]
        tile_info = self.level.tileset.get(str(tile_id))
        return tile_info is not None and tile_info.get("solid", False)

    def _wall_ahead(self, pixels_ahead):
        """True if a solid tile exists at torso height, pixels_ahead in front.
        Distinguishes walls (solid ahead, ground present) from gaps (no ground)."""
        check_x = self.mario.x + (self.mario.width * self.mario.scale) + pixels_ahead
        check_y = self.mario.y + (self.mario.height * self.mario.scale) * 0.5
        col = int(check_x // (self.level.tile_size * SCALE))
        row = int(check_y // (self.level.tile_size * SCALE))
        if not (0 <= row < self.level.height and 0 <= col < self.level.width):
            return False
        tile_id = self.level.grid[row][col]
        tile_info = self.level.tileset.get(str(tile_id))
        return tile_info is not None and tile_info.get("solid", False)

    def _find_gap_ahead(self, max_look=250):
        """Scan forward to find the nearest gap in the ground.
        Returns (gap_start_x, gap_end_x) in world pixels, or (None, None)."""
        tile_px = self.level.tile_size * SCALE
        gap_start = None
        gap_end = None
        x = self.mario.x
        end_x = x + max_look

        while x < end_x:
            if not self._ground_at_world_x(x):
                if gap_start is None:
                    gap_start = x
            else:
                if gap_start is not None:
                    gap_end = x
                    break
            x += tile_px

        if gap_start is not None and gap_end is None:
            gap_end = end_x

        return gap_start, gap_end

    def _ground_at_world_x(self, world_x):
        check_y = self.mario.y + (self.mario.height * self.mario.scale) + 5
        col = int(world_x // (self.level.tile_size * SCALE))
        row = int(check_y // (self.level.tile_size * SCALE))
        if not (0 <= row < self.level.height and 0 <= col < self.level.width):
            return False
        tile_id = self.level.grid[row][col]
        tile_info = self.level.tileset.get(str(tile_id))
        return tile_info is not None and tile_info.get("solid", False)

    def _estimate_landing_x(self, action=4):
        """Frame-by-frame physics simulation of jump landing position.
        Matches mario.py's exact integration: velocity_y += gravity * dt."""
        if not self.mario.on_ground:
            return None

        hold_frames = 6 if action == 3 else 18
        vx = self.mario.horizontal_speed
        vy = self.mario.jump_force
        gravity = GRAVITY_PER_SECOND
        dt = 1 / 60

        y = 0.0
        x = 0.0

        for frame in range(600):  # safety cap
            vy += gravity * dt
            y  += vy * dt
            x  += vx * dt

            # Apply jump cut after hold expires (mirrors mario.py behaviour)
            if frame >= hold_frames and vy < self.mario.min_jump_velocity:
                vy = self.mario.min_jump_velocity

            # Landed at or below launch height (skip first few frames)
            if frame > 5 and y >= 0:
                break

        return self.mario.x + x

    # ------------------------------------------------------------------
    def _get_state(self):
        gap_start, gap_end = self._find_gap_ahead(max_look=250)
        landing_x = self._estimate_landing_x(action=4)

        return np.array([
            self.mario.x,
            self.mario.y,
            self.mario.velocity_x,
            self.mario.velocity_y,
            float(self.mario.on_ground),
            # Ground sensors (foot level)
            float(self._ground_ahead(16)),
            float(self._ground_ahead(32)),
            float(self._ground_ahead(64)),
            float(self._ground_ahead(96)),
            float(self._ground_ahead(128)),
            # Wall sensors (torso level)
            float(self._wall_ahead(16)),
            float(self._wall_ahead(32)),
            float(self._wall_ahead(48)),
            float(self._wall_ahead(64)),
            float(self._wall_ahead(80)),
            # Trajectory
            landing_x if landing_x is not None else self.mario.x,
            gap_start  if gap_start  is not None else 0.0,
            gap_end    if gap_end    is not None else 0.0,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def _update_camera(self):
        if self.mario.x - self.camera_x > SCREEN_WIDTH // 2:
            self.camera_x = self.mario.x - SCREEN_WIDTH // 2
        level_pixel_width = self.level.width * self.level.tile_size * SCALE
        if self.camera_x > level_pixel_width - SCREEN_WIDTH:
            self.camera_x = level_pixel_width - SCREEN_WIDTH

    def render(self):
        if not self.render_mode:
            return
        self.clock.tick(60)
        self.screen.fill((92, 148, 252))
        self.level.draw(self.screen, 0, self.camera_x, SCALE)
        self.mario.draw(self.screen, self.sprites, self.camera_x, 0)
        pygame.display.flip()