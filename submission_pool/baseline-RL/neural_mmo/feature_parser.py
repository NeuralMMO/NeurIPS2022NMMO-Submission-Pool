from typing import Dict, Tuple

import numpy as np
from gym import spaces
from neurips2022nmmo import CompetitionConfig
from numpy import ndarray


class FeatureParser:
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5, 14, 15)  # lava, water, stone,
    spec = {
        "terrain":
        spaces.Box(low=0, high=15, shape=(15, 15), dtype=np.int64),
        "reachable":
        spaces.Box(low=0, high=1, shape=(15, 15), dtype=np.float32),
        "death_fog_damage":
        spaces.Box(low=0, high=1, shape=(15, 15), dtype=np.float32),
        "entity_population":
        spaces.Box(low=0, high=5, shape=(15, 15), dtype=np.int64),
        "self_entity":
        spaces.Box(low=0, high=1, shape=(1, 26), dtype=np.float32),
        "other_entity":
        spaces.Box(low=0, high=1, shape=(15, 26), dtype=np.float32),
        "va_move":
        spaces.Box(low=0, high=1, shape=(5, ), dtype=np.float32),
        "va_attack_target":
        spaces.Box(low=0, high=1, shape=(16, ), dtype=np.float32),
    }

    def parse(
        self,
        observations: Dict[int, Dict[str, ndarray]],
        step: int,
    ) -> Dict[str, ndarray]:
        ret = {}
        for agent_id in observations:
            terrain, death_fog_damage, population, reachable, va_move = self.parse_local_map(
                observations[agent_id], step)
            entity, va_target = self.parse_entity(observations[agent_id])
            self_entity = entity[:1, :]
            other_entity = entity[1:, :]
            ret[agent_id] = {
                "terrain": terrain,
                "death_fog_damage": death_fog_damage,
                "reachable": reachable,
                "entity_population": population,
                "self_entity": self_entity,
                "other_entity": other_entity,
                "va_move": va_move,
                "va_attack_target": va_target,
            }
        return ret

    def parse_local_map(
        self,
        observation: Dict[str, ndarray],
        step: int,
    ) -> Tuple[ndarray, ndarray]:
        tiles = observation["Tile"]["Continuous"]
        entities = observation["Entity"]["Continuous"]
        terrain = np.zeros(shape=self.spec["terrain"].shape,
                           dtype=self.spec["terrain"].dtype)
        death_fog_damage = np.zeros(shape=self.spec["death_fog_damage"].shape,
                                    dtype=self.spec["death_fog_damage"].dtype)
        population = np.zeros(shape=self.spec["entity_population"].shape,
                              dtype=self.spec["entity_population"].dtype)
        va = np.ones(shape=self.spec["va_move"].shape,
                     dtype=self.spec["va_move"].dtype)

        # terrain, death_fog
        R, C = tiles[0, 2:4]
        for tile in tiles:
            absolute_r, absolute_c = tile[2:4]
            relative_r, relative_c = int(absolute_r - R), int(absolute_c - C)
            terrain[relative_r, relative_c] = int(tile[1])
            dmg = self.compute_death_fog_damage(absolute_r, absolute_c, step)
            death_fog_damage[relative_r, relative_c] = dmg / 100.0

        # entity population map
        P = entities[0, 6]
        for e in entities:
            if e[0] == 0: break
            absolute_r, absolute_c = e[7:9]
            relative_r, relative_c = int(absolute_r - R), int(absolute_c - C)
            if e[6] == P:
                p = 1
            elif e[6] >= 0:
                p = 2
            elif e[6] < 0:
                p = abs(e[6]) + 2
            population[relative_r, relative_c] = p

        # reachable area
        reachable = self.gen_reachable_map(terrain)

        # valid move
        for i, (r, c) in enumerate(self.NEIGHBOR):
            if terrain[r, c] in self.OBSTACLE:
                va[i + 1] = 0

        return terrain, death_fog_damage, population, reachable, va

    def parse_entity(
        self,
        observation: Dict[str, ndarray],
        max_size: int = 16,
    ) -> Tuple[ndarray, ndarray]:
        cent = CompetitionConfig.MAP_CENTER // 2
        entities = observation["Entity"]["Continuous"]
        va = np.zeros(shape=self.spec["va_attack_target"].shape,
                      dtype=self.spec["va_attack_target"].dtype)
        va[0] = 1.0

        entities_list = []
        P, R, C = entities[0, 6:9]
        for i, e in enumerate(entities[:max_size]):
            if e[0] == 0: break
            # attack range
            p, r, c = e[6:9]
            if p != P and abs(R - r) <= 3 and abs(C - c) <= 3:
                va[i] = 1
            # population
            population = [0 for _ in range(5)]
            if p == P:
                population[0] = 1
            elif p >= 0:
                population[1] = 1
            elif p < 0:
                population[int(abs(p)) + 1] = 1
            entities_list.append(
                np.array(
                    [
                        float(e[2] == 0),  # attacked
                        e[3] / 10.0,  # level
                        e[4] / 10.0,  # item_level
                        (r - 16) / 128.0,  # r
                        (c - 16) / 128.0,  # c
                        (r - 16 - cent) / 128.0,  # delta_r
                        (c - 16 - cent) / 128.0,  # delta_c
                        e[9] / 100.0,  # damage
                        e[10] / 1024.0,  # alive_time
                        e[12] / 100.0,  # gold
                        e[13] / 100.0,  # health
                        e[14] / 100.0,  # food
                        e[15] / 100.0,  # water
                        e[16] / 10.0,  # melee
                        e[17] / 10.0,  # range
                        e[18] / 10.0,  # mage
                        e[19] / 10.0,  # fishing
                        e[20] / 10.0,  # herbalism
                        e[21] / 10.0,  # prospecting
                        e[22] / 10.0,  # carving
                        e[23] / 10.0,  # alchmy
                        *population,
                    ],
                    dtype=np.float32))
        if len(entities_list) < max_size:
            entities_list.extend([
                np.zeros_like(entities_list[0])
                for _ in range(max_size - len(entities_list))
            ])
        return np.asarray(entities_list), va

    @staticmethod
    def compute_death_fog_damage(r: int, c: int, step: int) -> float:
        C = CompetitionConfig
        if step < C.PLAYER_DEATH_FOG:
            return 0
        r, c = r - 16, c - 16
        cent = C.MAP_CENTER // 2
        # Distance from center of the map
        dist = max(abs(r - cent), abs(c - cent))
        if dist > C.PLAYER_DEATH_FOG_FINAL_SIZE:
            time_dmg = C.PLAYER_DEATH_FOG_SPEED * (step - C.PLAYER_DEATH_FOG +
                                                   1)
            dist_dmg = dist - cent
            dmg = max(0, dist_dmg + time_dmg)
        else:
            dmg = 0
        return dmg

    def gen_reachable_map(self, terrain: ndarray) -> ndarray:
        """
        grid: M * N
            1: passable
            0: unpassable
        """
        from collections import deque
        M, N = terrain.shape
        passable = ~np.isin(terrain, self.OBSTACLE)
        reachable = np.zeros_like(passable)
        visited = np.zeros_like(passable)
        q = deque()
        start = M // 2, N // 2
        q.append(start)
        visited[start[0], start[1]] = 1
        while q:
            cur_r, cur_c = q.popleft()
            reachable[cur_r, cur_c] = 1
            for (dr, dc) in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
                r, c = cur_r + dr, cur_c + dc
                if not (0 <= r < M and 0 <= c < N):
                    continue
                if not visited[r, c] and passable[r, c]:
                    q.append((r, c))
                visited[r, c] = 1
        return reachable
