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

        # Jump hold state — env simulates button hold internally
        self._jump_frames_left = 0   # how many frames to keep boosting
        self._jump_holding = False

        # 0: idle  1: walk right  2: walk left
        # 3: short jump+right (~6 frames)
        # 4: full jump+right  (~18 frames, clears wide gaps)
        # 5: full jump only   (no horizontal)
        self.action_space = spaces.Discrete(6)

        # Observation (18 values):
        #   0-1:   mario x, y
        #   2-3:   velocity_x, velocity_y (normalised)
        #   4:     on_ground
        #   5-9:   ground sensors at 16,32,64,96,128 px ahead
        #   10-14: wall sensors at 16,32,48,64,80 px ahead (at torso height)
        #   15:    estimated landing_x
        #   16:    gap_start_x  (0 if none)
        #   17:    gap_end_x    (0 if none)
        self.observation_space = spaces.Box(
            low=-100_000, high=100_000,
            shape=(18,), dtype=np.float32
        )

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

    def reset(self):
        self.level = Level("levels/1-1.json", "tileset.json")
        self.mario = Mario(x=100, y=0, scale=SCALE)
        self.camera_x = 0
        self.prev_x = self.mario.x
        self._jump_frames_left = 0
        self._jump_holding = False
        return self._get_state()

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
        if self.mario.on_ground:
            reward += np.clip(delta_x, -8, 8) * 0.15
        else:
            reward += np.clip(delta_x, -8, 8) * 0.04

        if delta_x < 0:
            reward -= 0.08

        # ── 2. Grounded bonus / air tax ───────────────────────────────
        if self.mario.on_ground:
            reward += 0.03
        else:
            reward -= 0.015

        # ── 3. Wall collision penalty ─────────────────────────────────
        # If Mario barely moved while trying to go right, he hit a wall
        if action in [1, 3, 4] and delta_x < 1.0 and not self.mario.on_ground is False:
            reward -= 0.1

        # ── 4. Jump evaluation (fires once on jump frame) ─────────────
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
                    # Bonus scales with gap width — harder gaps earn more
                    gap_width = gap_end - gap_start
                    reward += 0.5 + min(gap_width / 200.0, 0.5)
                else:
                    reward -= 0.4   # jumped but arc won't clear it

            elif has_wall and not has_gap:
                reward -= 0.5   # jumped into a wall — bad

            elif not has_gap and not has_wall:
                reward -= 0.25  # unnecessary jump on flat ground

            # If both gap and wall: ambiguous, small penalty to discourage
            elif has_gap and has_wall:
                reward -= 0.1

        # ── 5. Air button spam ────────────────────────────────────────
        if action in [3, 4, 5] and not was_on_ground:
            reward -= 0.15

        # ── 6. Death / completion ─────────────────────────────────────
        floor_y = self.level.height * self.level.tile_size * SCALE
        if self.mario.y >= floor_y:
            reward -= 150
            done = True

        level_pixel_width = self.level.width * self.level.tile_size * SCALE
        if self.mario.x >= level_pixel_width - 50:
            reward += 300
            done = True

        reward -= 0.005  # tiny living penalty

        self.prev_x = self.mario.x
        return self._get_state(), reward, done, {}

    def _apply_action(self, action, was_on_ground):
        """
        Actions 3/4/5 initiate a jump with a held duration.
        The env ticks down _jump_frames_left each step, simulating
        held Z — mario.py cuts the arc short if velocity_y is released.
        """
        self.mario.velocity_x = 0

        # Tick down any active jump hold
        if self._jump_frames_left > 0:
            self._jump_frames_left -= 1
            # Keep upward velocity boosted (simulate held jump)
            # mario.py applies min_jump_velocity cut when Z is released,
            # so as long as we keep calling with jump held, we get full arc
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
            # Short hop right — jump + right, hold for ~6 frames
            self.mario.velocity_x = self.mario.horizontal_speed
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 6
                self._jump_holding = True

        elif action == 4:
            # Full jump right — hold for ~18 frames → maximum arc
            self.mario.velocity_x = self.mario.horizontal_speed
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 18
                self._jump_holding = True

        elif action == 5:
            # Full jump no horizontal (for jumping over walls in place)
            if was_on_ground:
                self.mario.velocity_y = self.mario.jump_force
                self._jump_frames_left = 18
                self._jump_holding = True

        # If we're mid-hold, keep applying horizontal for actions 3/4
        if self._jump_holding and not was_on_ground:
            if action in [3, 4]:
                self.mario.velocity_x = self.mario.horizontal_speed

        # Simulate held jump: prevent mario.py's min_jump_velocity cut
        # by keeping velocity_y below min_jump_velocity threshold while holding
        if self._jump_holding and self.mario.velocity_y < 0:
            # Don't cut the jump arc — leave velocity_y alone
            pass
        elif not self._jump_holding and self.mario.velocity_y < self.mario.min_jump_velocity:
            # Simulate releasing jump button — apply the cut
            self.mario.velocity_y = self.mario.min_jump_velocity

    def _ground_ahead(self, pixels_ahead):
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
        """
        Check for a solid tile at torso height ahead of Mario.
        This distinguishes walls (jump over them) from gaps (jump across).
        """
        check_x = self.mario.x + (self.mario.width * self.mario.scale) + pixels_ahead
        # Check at mid-body height
        check_y = self.mario.y + (self.mario.height * self.mario.scale) * 0.5
        col = int(check_x // (self.level.tile_size * SCALE))
        row = int(check_y // (self.level.tile_size * SCALE))
        if not (0 <= row < self.level.height and 0 <= col < self.level.width):
            return False
        tile_id = self.level.grid[row][col]
        tile_info = self.level.tileset.get(str(tile_id))
        return tile_info is not None and tile_info.get("solid", False)

    def _find_gap_ahead(self, max_look=250):
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
        """
        Predict landing x for the given action's hold duration.
        Uses mario.py's exact physics: velocity_y += gravity * dt each frame.
        """
        if not self.mario.on_ground:
            return None

        gravity = GRAVITY_PER_SECOND
        vy = self.mario.jump_force
        vx = self.mario.horizontal_speed

        # Simulate frame by frame until we return to launch height
        # This accounts for the min_jump_velocity cut on short hops
        hold_frames = 6 if action == 3 else 18
        y = 0.0
        vy_sim = vy
        x = 0.0
        dt = 1 / 60

        for frame in range(500):  # safety limit
            vy_sim += gravity * dt
            y += vy_sim * dt
            x += vx * dt

            # Apply jump cut after hold expires
            if frame >= hold_frames and vy_sim < self.mario.min_jump_velocity:
                vy_sim = self.mario.min_jump_velocity

            # Landed back at or below launch height
            if frame > 5 and y >= 0:
                break

        return self.mario.x + x

    def _get_state(self):
        gap_start, gap_end = self._find_gap_ahead(max_look=250)
        landing_x = self._estimate_landing_x(action=4)  # always report full-jump estimate

        return np.array([
            self.mario.x,
            self.mario.y,
            self.mario.velocity_x,
            self.mario.velocity_y,
            float(self.mario.on_ground),
            # Ground sensors
            float(self._ground_ahead(16)),
            float(self._ground_ahead(32)),
            float(self._ground_ahead(64)),
            float(self._ground_ahead(96)),
            float(self._ground_ahead(128)),
            # Wall sensors
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