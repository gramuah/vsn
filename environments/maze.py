import copy

import numpy as np
from gym import spaces
from gym.spaces import Dict
from gym.spaces import Box as SpaceBox
from gym_miniworld.entity import Agent
from gym_miniworld.entity import Box
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS
from utils.custom_networks import clip


class PyMaze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
            self,
            num_rows=8,
            num_cols=8,
            room_size=3,
            max_steps=None,
            forward_step=0.7,
            turn_step=45,
            use_clip=True,
            domain_rand=False,
            sparse_reward=True,
            **kwargs):

        self.sparse_reward = sparse_reward
        self.use_clip = use_clip
        self.num_rows = num_rows
        self.forward_step = forward_step
        self.turn_step = turn_step
        self.max_steps = max_steps
        self.num_cols = num_cols
        self.previous_measure = None
        self.room_size = room_size
        self.gap_size = 0.25
        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", forward_step)
        params.set("turn_step", turn_step)

        self.observation_space = SpaceBox(
            low=0, high=255, shape=(60, 80, 3), dtype=np.uint8
        )

        if self.use_clip:
            obs_dict = {"rgb": self.observation_space}
            self.clipResNet = clip.ResNetCLIPEncoder(Dict(obs_dict), is_habitat=False)

        super().__init__(
            max_episode_steps=max_steps or num_rows * num_cols * 24,
            domain_rand=domain_rand,
            params=params,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0, 1), (0, -1), (-1, 0), (1, 0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        X = (self.num_cols - 0.5) * self.room_size + (self.num_cols - 1) * self.gap_size
        Z = (self.num_rows - 0.5) * self.room_size + (self.num_rows - 1) * self.gap_size
        self.box = self.place_entity(Box(color='red'), pos=np.array([X, 0, Z]))

        X = 0.5 * self.room_size
        Z = 0.5 * self.room_size
        self.place_entity(self.agent, pos=np.array([X, 0, Z]), dir=0)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.use_clip:
            obs = self.clipResNet.forward(obs)
            obs = obs.squeeze()

        reward += self._reward()

        if self.near(self.box):
            done = True

        return obs, reward, done, info

    def _reward(self):
        """
        Default sparse reward computation
        """
        reward = 0

        if self.sparse_reward and self.near(self.box):
            reward += 1.0 - 0.2 * (self.step_count / self.max_episode_steps)
        elif not self.sparse_reward:
            # Default habitat reward
            reward += -0.01  # slack reward
            current_measeure = np.linalg.norm(self.box.pos - self.agent.pos)

            reward += self.previous_measure - current_measeure
            self.previous_measure = current_measeure

            if self.near(self.box):
                reward += 1.0 - 0.2 * (self.step_count / self.max_episode_steps)

        return reward

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'sky_color',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max('forward_step')

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min([r.min_x for r in self.rooms])
        self.max_x = max([r.max_x for r in self.rooms])
        self.min_z = min([r.min_z for r in self.rooms])
        self.max_z = max([r.max_z for r in self.rooms])

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Get the distance between target and agent

        self.previous_measure = np.linalg.norm(self.box.pos - self.agent.pos)

        # Generate the first camera image
        obs = self.render_obs()
        if self.use_clip:
            obs = self.clipResNet.forward(obs)
            obs = obs.squeeze()
        # Return first observation
        return obs

    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     deepcopy = cls.__new__(cls)
    #     memo[id(self)] = deepcopy
    #
    #     # This attributes don't support deepcopy directly
    #     text_label_copy = self.__dict__["text_label"]
    #     shadow_window_copy = self.__dict__["shadow_window"]
    #     rooms_copy = self.__dict__["rooms"]
    #     setattr(deepcopy, "text_label", text_label_copy)
    #     setattr(deepcopy, "shadow_window", shadow_window_copy)
    #     setattr(deepcopy, "rooms", rooms_copy)
    #
    #     # For the rest apply standard deepcopy
    #     for k, v in self.__dict__.items():
    #         if k not in ["text_label", "shadow_window", "rooms"]:
    #             setattr(deepcopy, k, copy.deepcopy(v, memo))
    #     return deepcopy

    def __getstate__(self):
        """
        See `Object.__getstate__.
        Returns:
            dict: The instance’s dictionary to be pickled.
        """
        return dict(num_rows=self.num_rows,
                    num_cols=self.num_cols,
                    room_size=self.room_size,
                    max_steps=self.max_steps,
                    forward_step=self.forward_step,
                    turn_step=self.turn_step,
                    use_clip=self.use_clip,
                    domain_rand=self.domain_rand,
                    sparse_reward=self.domain_rand)

    def __setstate__(self, state):
        """
        See `Object.__setstate__.
        Args:
            state (dict): Unpickled state of this object.
        """
        self.__init__(num_rows=state['num_rows'],
                      num_cols=state['num_cols'],
                      room_size=state['room_size'],
                      max_steps=state['max_steps'],
                      forward_step=state['forward_step'],
                      turn_step=state['turn_step'],
                      use_clip=state['use_clip'],
                      domain_rand=state['domain_rand'],
                      sparse_reward=state['domain_rand'])
