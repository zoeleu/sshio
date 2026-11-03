"""GPU-resident training arenas; authoritative validation remains in Go.

Physics uses the same tick/substep constants and ordered body constraint. Float32
arithmetic, finite explicit capacities, GPU spawn sampling, and deterministic
multi-contact tie breaking are documented differences from the Go server.
"""
import math
import torch
import triton
from gpu_kernels import (move_kernel, collision_kernel, resolve_kernel,
                         resize_kernel, allocate_kernel, release_kernel,
                         pickup_kernel, rays_kernel, spawn_kernel, food_kernel)


class GPUSim:
    def __init__(self, arenas, agents, horizon, seed, device="cuda",
                 max_segments=512, max_pellets=16384, curriculum=True, stage=0,
                 personalities=False):
        if not 1 <= arenas <= 64 or not 1 <= agents <= 64 or arenas * agents > 2048:
            raise ValueError("GPU simulator needs 1..64 arenas/agents and <=2048 worms")
        if torch.device(device).type != "cuda":
            raise ValueError("GPUSim requires a GPU; there is no CPU simulation fallback")
        if max_segments < 80 or max_pellets < 64:
            raise ValueError("GPU capacities need at least 80 segments and 64 pellets")
        self.a, self.n, self.s, self.p = arenas, agents, max_segments, max_pellets
        self.personalities = bool(personalities)
        self.obs_dim = 692 if self.personalities else 688
        self.move_block = 8
        self.horizon, self.device = horizon, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.curriculum, self.stage, self.episode = curriculum, stage, -1
        self.turns = torch.tensor([-.9, -.4, -.15, 0, .15, .4, .9], device=self.device)
        self.segment = torch.arange(self.s, device=self.device)
        self.worm = torch.arange(self.n, device=self.device)
        self.pellet_index = torch.arange(self.p, device=self.device)
        self.same = torch.eye(self.n, device=self.device, dtype=torch.bool)[None]
        if self.personalities:
            slot_p = ((torch.arange(self.a, device=self.device)[:, None] + self.worm[None]) % 4).long()
            self.personality_id = slot_p
            self.personality_onehot = torch.nn.functional.one_hot(slot_p, num_classes=4).float().reshape(self.a * self.n, 4)
            # Personalities: 0=Aggressive, 1=Greedy, 2=Conservative, 3=Afraid
            self.w_score = torch.tensor([0.15, 0.55, 0.30, 0.25], device=self.device)[slot_p]
            self.w_kill = torch.tensor([45.0, 2.0, 3.0, 1.0], device=self.device)[slot_p]
            self.w_death = torch.tensor([6.0, 9.0, 18.0, 18.0], device=self.device)[slot_p]
            self.w_fear = torch.tensor([0.0, 0.0, 0.0, 0.15], device=self.device)[slot_p]
        maximum_radius = max(300 * math.sqrt(self.n / 15),
                             90 * math.sqrt(max(1, self.n / 4)) * 1.2 * 1.4)
        self.cells = 2 * math.ceil((maximum_radius + self.s * .65 + 64) / 12) + 4
        self.overflow = torch.zeros((), dtype=torch.int32, device=self.device)
        self.last_observation = None
        self.reset()

    def random(self, *shape):
        return torch.rand(shape, generator=self.generator, device=self.device)

    def reset(self):
        self.episode += 1
        self.age, self.tick = 0, 0
        a, n, s, p, d = self.a, self.n, self.s, self.p, self.device
        choices = (("forage", "border", "arena"),
                   ("forage", "border", "cutoff", "escape", "chase", "arena"),
                   ("forage", "border", "cutoff", "escape", "chase", "arena", "sparse", "dense", "production"))
        options = choices[self.stage]
        self.scenarios = [options[(i + self.episode) % len(options)] if self.curriculum else "arena" for i in range(a)]
        scale = max(1, n / 4)
        self.radius = torch.full((a,), 90 * math.sqrt(scale), device=d)
        self.pellet_target = torch.full((a,), int(320 * scale), device=d, dtype=torch.int32)
        if self.curriculum:
            self.radius *= .8 + .4 * self.random(a)
            self.pellet_target = (self.pellet_target * (.6 + .8 * self.random(a))).int()
            profiles = torch.tensor([{"sparse": 1, "dense": 2, "production": 3}.get(x, 0) for x in self.scenarios], device=d)
            self.radius *= torch.where(profiles == 1, 1.4, torch.where(profiles == 2, .75, 1.))
            self.radius = torch.where(profiles == 3, 300 * math.sqrt(n / 15), self.radius)
            self.pellet_target = torch.where(profiles == 3, 3500 * n // 15, self.pellet_target)
        if p < math.ceil(max(320 * scale * 1.4, 3500 * n / 15)):
            raise ValueError("max_pellets is smaller than the arena's food target")
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        grid = torch.stack(((self.worm % cols).float() - (cols - 1) / 2,
                            (self.worm // cols).float() - (rows - 1) / 2), -1)
        head = grid[None] * (self.radius[:, None, None] * 1.1 / cols)
        head = head + (self.random(a, n, 2) - .5) * 4
        self.heading = self.random(a, n) * (2 * math.pi)
        self.score = torch.full((a, n), 12., device=d)
        if self.curriculum:
            forage = torch.tensor([x == "forage" for x in self.scenarios], device=d)[:, None]
            self.score += torch.where(forage, 0., self.random(a, n) * 45)
            head = self._arrange(head)
        self.target = self.heading.clone()
        self.length = self.score.int() + 8
        self.alive = torch.ones((a, n), dtype=torch.bool, device=d)
        forward = torch.stack((self.heading.cos(), self.heading.sin()), -1)
        self.body = (head[:, None] - forward[:, None] * self.segment[None, :, None, None] * .65).contiguous()
        self.boost = torch.zeros_like(self.alive)
        self.kills = torch.zeros((a, n), device=d, dtype=torch.int32)
        self.mutual = torch.zeros_like(self.kills)
        self.food, self.spent = torch.zeros_like(self.score), torch.zeros_like(self.score)
        self.ids = self.worm[None].expand(a, -1).int().clone() + 1
        self.next_id = torch.full((a,), n, device=d, dtype=torch.int32)
        self.drop_pos = torch.zeros((a, n, 2), device=d)
        self.drop_mask = torch.zeros_like(self.alive)
        self.dead, self.tick_dead = torch.zeros_like(self.alive), torch.zeros_like(self.alive)
        self.killer = torch.full_like(self.kills, -1)
        self.tick_killer = torch.full_like(self.kills, -1)
        self.killed_size = torch.zeros_like(self.score)
        self.corpse_eaten = torch.zeros_like(self.score)
        self.owner = torch.full((a, p), -1, dtype=torch.int32, device=d)
        self.pellet_pos = self._random_pellets()
        self.pellet_value = torch.ones((a, p), device=d)
        self.pellet_killer = torch.full((a, p), -1, dtype=torch.int32, device=d)
        self.active = self.pellet_index[None] < self.pellet_target[:, None]
        self.free_list = ((self.pellet_index[None] + self.pellet_target[:, None]) % p).int()
        self.free_count = (p - self.pellet_target).int()
        self.loop_origin, self.loop_heading = head.clone(), self.heading.clone()
        self.loop_turn = torch.zeros_like(self.score)
        self.loop_steps = torch.zeros_like(self.kills)
        self.prev_heading = self.heading.clone()
        self.turn_rate = torch.zeros_like(self.score)
        self.boosting = torch.zeros_like(self.alive)
        self.stack = torch.zeros((a, n, 4, 172), device=d)
        self.stack_ready = False
        self._stack_reset_mask = torch.zeros_like(self.alive)
        self.last_observation = torch.zeros((a * n, self.obs_dim), device=d)
        if self.curriculum:
            # Food trails use the same pool as ordinary/death/boost pellets.
            trail = head[:, :, None] + forward[:, :, None] * (torch.arange(12, device=d) + 8)[None, None, :, None]
            mask = torch.tensor([x == "forage" for x in self.scenarios], device=d)[:, None, None].expand(a, n, 12)
            self._allocate(trail.reshape(a, -1, 2), torch.ones((a, n * 12), device=d), mask.reshape(a, -1))
        return self.state()

    def _arrange(self, head):
        # Metadata chooses profiles on the host only at episode boundaries;
        # positions, angles, scores, and all simulation state are device tensors.
        for arena, scenario in enumerate(self.scenarios):
            if scenario == "border":
                angle = self.random(1) * (2 * math.pi) + self.worm * (2 * math.pi / self.n)
                head[arena] = torch.stack((angle.cos(), angle.sin()), -1) * (self.radius[arena] - 10)
                self.heading[arena] = angle
                self.score[arena] = 20
            elif scenario in ("cutoff", "escape", "chase"):
                pairs = self.n // 2
                cols, rows = math.ceil(math.sqrt(max(1, pairs))), math.ceil(max(1, pairs) / math.ceil(math.sqrt(max(1, pairs))))
                pair = torch.arange(pairs, device=self.device)
                center = torch.stack((pair % cols - (cols - 1) / 2,
                                      pair // cols - (rows - 1) / 2), -1) * (self.radius[arena] * 1.1 / cols)
                angle = self.random(pairs) * (2 * math.pi)
                p0, p1, h1, scores = (0, -10), (-10, 0), 0., (30., 30.)
                if scenario == "escape":
                    p0, p1, scores = (0, -7), (-7, 0), (20., 70.)
                if scenario == "chase":
                    p0, p1, h1, scores = (0, -12), (0, 10), math.pi / 2, (45., 20.)
                for side, pos, heading in ((0, p0, math.pi / 2), (1, p1, h1)):
                    rotate = torch.stack((pos[0] * angle.cos() - pos[1] * angle.sin(),
                                          pos[0] * angle.sin() + pos[1] * angle.cos()), -1)
                    head[arena, side:2 * pairs:2] = center + rotate
                    self.heading[arena, side:2 * pairs:2] = angle + heading
                    self.score[arena, side:2 * pairs:2] = scores[side]
        return head

    def _random_pellets(self):
        angle = self.random(self.a, self.p) * (2 * math.pi)
        radius = self.random(self.a, self.p).sqrt() * (self.radius[:, None] - 2)
        return torch.stack((angle.cos() * radius, angle.sin() * radius), -1)

    def _allocate(self, positions, values, mask, killers=None):
        if killers is None:
            killers = torch.full((self.a, mask.shape[1]), -1, dtype=torch.int32, device=self.device)
        positions, values, mask, killers = positions.contiguous(), values.contiguous(), mask.contiguous(), killers.contiguous()
        ranks = mask.cumsum(1, dtype=torch.int32)
        requests = mask.shape[1]
        allocate_kernel[(self.a, triton.cdiv(requests, 256))](
            self.free_list, self.free_count, ranks, mask, positions, values, killers,
            self.pellet_pos, self.pellet_value, self.pellet_killer, self.active, self.overflow,
            self.p, requests, 256)
        self.free_count.sub_(mask.sum(1).int()).clamp_(min=0)

    def _release(self, mask):
        ranks = mask.cumsum(1, dtype=torch.int32)
        release_kernel[(self.a, triton.cdiv(self.p, 256))](
            self.free_list, self.free_count, mask, ranks, self.active, self.p, 256)
        self.free_count.add_(mask.sum(1).int())

    def _tick(self):
        self.tick += 1
        self.tick_dead.zero_()
        self.tick_killer.fill_(-1)
        self.boosting.copy_(self.boost & (self.score > 12))
        maximum = self.length.amax(1)
        for substep in range(4):
            self.boosting.copy_(self.boost & (self.score > 12))
            move_kernel[(self.a, triton.cdiv(self.n, self.move_block))](
                self.body, self.heading, self.target, self.score, self.length,
                self.alive, self.boost, self.spent, self.drop_pos, self.drop_mask,
                self.n, self.s, substep == 0, self.tick % 4 == 0, self.move_block, num_warps=1)
            collision_kernel[(self.a, self.n)](
                self.body, self.length, self.score, self.alive, self.ids, self.radius, maximum,
                self.dead, self.killer, self.n, self.s, 256)
            resolve_kernel[(triton.cdiv(self.a * self.n, 128),)](
                self.alive, self.dead, self.killer, self.score, self.boosting, self.kills, self.mutual,
                self.tick_dead, self.killed_size, self.tick_killer, self.a * self.n, self.n, 128)
        turned = self.heading - self.prev_heading
        self.turn_rate.copy_(((turned + math.pi) % (2 * math.pi) - math.pi) / 0.18)
        self.turn_rate.clamp_(-1, 1)
        self.prev_heading.copy_(self.heading)
        corpse_pos = self.body[:, ::3].reshape(self.a, -1, 2)
        corpse_mask = ((self.segment[::3][None, :, None] < self.length[:, None]) & self.tick_dead[:, None]).reshape(self.a, -1)
        positions = torch.cat((self.drop_pos, corpse_pos), 1)
        values = torch.cat((torch.full_like(self.score, .4), torch.full_like(corpse_mask, 2., dtype=torch.float32)), 1)
        corpse_killer = self.tick_killer[:, None, :].expand(-1, len(self.segment[::3]), -1).reshape(self.a, -1)
        drop_killer = torch.full((self.a, self.n), -1, dtype=torch.int32, device=self.device)
        killers = torch.cat((drop_killer, corpse_killer), 1)
        mask = torch.cat((self.drop_mask, corpse_mask), 1)
        self._allocate(positions, values, mask, killers)
        pickup_kernel[(self.a, triton.cdiv(self.p, 64))](
            self.body, self.score, self.alive, self.ids, self.pellet_pos, self.active, self.owner,
            self.n, self.s, self.p, triton.next_power_of_2(self.n), 64)
        consumed = self.owner >= 0
        gain = torch.zeros_like(self.score).scatter_add_(1, self.owner.clamp_min(0).long(),
                                                       self.pellet_value * consumed)
        eaters = self.owner.clamp_min(0).long()
        from_own_kill = consumed & (self.pellet_killer >= 0) & (self.pellet_killer.long() == eaters)
        eaten_from_kill = torch.zeros_like(self.score).scatter_add_(1, eaters, self.pellet_value * from_own_kill)
        self.corpse_eaten.add_(eaten_from_kill)
        eff = 1.0 / (1.0 + (self.score - 60.0).clamp_min(0.0) / 120.0)
        self.score.add_(gain * eff)
        self.food.add_(gain)
        decay = torch.where(self.alive & (self.score > 120.0), 0.002 * (self.score - 120.0), 0.0)
        self.score.sub_(decay)
        self._release(consumed)
        resize_kernel[(self.a,)](self.body, self.score, self.length, self.alive,
                                self.overflow, self.n, self.s, triton.next_power_of_2(self.n))
        worm_mass = ((self.score - 12.0).clamp_min(0.0) * self.alive).sum(1)
        pellet_mass = (self.pellet_value * self.active).sum(1)
        total_mass = worm_mass + pellet_mass
        active_count = self.active.sum(1)
        max_by_count = (self.pellet_target - active_count).clamp_min(0)
        max_by_mass = (self.pellet_target.float() - total_mass).floor().long().clamp_min(0)
        floor_needed = (self.pellet_target // 4 - active_count).clamp_min(0)
        needed = torch.maximum(floor_needed, torch.minimum(max_by_count, max_by_mass)).clamp_min(0)
        self._allocate(self._random_pellets(), torch.ones_like(self.pellet_value),
                       self.pellet_index[None] < needed[:, None])

    def observe_single(self):
        a, n, p, d = self.a, self.n, self.p, self.device
        head, h = self.body[:, 0], self.heading
        forward = torch.stack((h.cos(), h.sin()), -1)
        side = torch.stack((-h.sin(), h.cos()), -1)
        obs = torch.zeros((a, n, 172), device=d)
        obs[:, :, 0] = (self.score / 100).clamp(-1, 1)
        obs[:, :, 1] = (torch.log((self.score / 12).clamp_min(1)) / 3).clamp(-1, 1)
        obs[:, :, 2] = self.score > 12
        norm = head.norm(dim=-1)
        obs[:, :, 3] = ((self.radius[:, None] - norm) / 40).clamp(-1, 1)
        radial = (norm / self.radius[:, None]).clamp_max(1)
        pull = ((radial - 0.60) / 0.40).clamp(0, 1)
        inward = -head / norm.clamp_min(1.e-8)[:, :, None]
        obs[:, :, 4], obs[:, :, 5] = (inward * forward).sum(-1) * pull, (inward * side).sum(-1) * pull
        speed = torch.where(self.boosting, 1.6, 0.8)
        obs[:, :, 6] = speed / 1.6
        obs[:, :, 7] = self.boosting.float()
        obs[:, :, 8] = self.turn_rate.clamp(-1, 1)
        obs[:, :, 9] = (self.turn_rate * 0.2).clamp(-1, 1)
        obs[:, :, 10] = self.boosting.float()
        obs[:, :, 11] = (self.length.float() / 120).clamp_max(1)
        obs[:, :, 13] = self.score - self.score.floor()
        obs[:, :, 14] = ((self.score - 12) / 50).clamp(-1, 1)
        obs[:, :, 15] = radial
        cell = ((self.pellet_pos + 0.001) / 12).floor().long() + self.cells // 2
        cell = cell.clamp(0, self.cells - 1)
        cell_index = cell[:, :, 0] * self.cells + cell[:, :, 1]
        patches = torch.zeros((a, self.cells * self.cells), device=d).scatter_add_(1, cell_index, self.pellet_value * self.active)
        richness = patches.gather(1, cell_index)
        utility = torch.empty((a, n, p), device=d)
        sector_food = torch.empty((a, n, 24), device=d)
        food_kernel[(a, n)](self.body, h, self.radius, self.pellet_pos, self.pellet_value,
                            self.active, richness, utility, sector_food, n, self.s, p, 256)
        # Food targets: top-4 pellets by the kernel's utility, best first.
        top_value, top_index = utility.topk(4, dim=-1)
        arena_index = torch.arange(a, device=d)[:, None, None]
        target_pos = self.pellet_pos[arena_index, top_index]
        dist = (target_pos - head[:, :, None]).norm(dim=-1)
        target_pos = torch.where((target_pos.norm(dim=-1) > (self.radius[:, None, None] - 10))[..., None],
                                 (head * 0.5)[:, :, None], target_pos)
        delta = target_pos - head[:, :, None]
        pellet_value = self.pellet_value[arena_index, top_index]
        pellet_rich = richness[arena_index, top_index]
        valid = top_value.isfinite()
        food = torch.stack((((delta * forward[:, :, None]).sum(-1) / 40).clamp(-1, 1),
                            ((delta * side[:, :, None]).sum(-1) / 40).clamp(-1, 1),
                            (dist / 60).clamp_max(1),
                            ((pellet_value + pellet_rich.clamp_max(15) * 0.3) / 18).clamp_max(1)), -1)
        obs[:, :, 16:32] = torch.where(valid[..., None], food, 0.).reshape(a, n, 16)
        # Rivals: four nearest, with velocity/heading/boost/turn state.
        rivals = head[:, None] - head[:, :, None]
        rival_distance = rivals.norm(dim=-1).masked_fill(self.same | ~self.alive[:, None], float("inf"))
        sorted_dist, order = rival_distance.sort(dim=-1)
        if order.shape[-1] < 4:
            pad = 4 - order.shape[-1]
            order = torch.cat((order, torch.zeros((a, n, pad), device=d, dtype=torch.long)), -1)
            sorted_dist = torch.cat((sorted_dist, torch.full((a, n, pad), float("inf"), device=d)), -1)
        order4, dist4 = order[:, :, :4], sorted_dist[:, :, :4]
        exists = dist4.isfinite()
        relative = rivals.gather(2, order4[..., None].expand(-1, -1, -1, 2))
        rival_heading = self.heading[:, None].expand(-1, n, -1).gather(2, order4)
        rival_score = self.score[:, None].expand(-1, n, -1).gather(2, order4)
        rival_boosting = self.boosting[:, None].expand(-1, n, -1).gather(2, order4)
        rival_turn = self.turn_rate[:, None].expand(-1, n, -1).gather(2, order4)
        rival_speed = torch.where(rival_boosting, 1.6, 0.8)
        rival_vel = torch.stack((rival_heading.cos(), rival_heading.sin()), -1) * rival_speed[..., None]
        rel_heading = rival_heading - h[:, :, None]
        rival_block = torch.stack((
            ((relative * forward[:, :, None]).sum(-1) / 40).clamp(-1, 1),
            ((relative * side[:, :, None]).sum(-1) / 40).clamp(-1, 1),
            ((rival_vel * forward[:, :, None]).sum(-1) / 1.6).clamp(-1, 1),
            ((rival_vel * side[:, :, None]).sum(-1) / 1.6).clamp(-1, 1),
            rel_heading.cos(), rel_heading.sin(),
            (rival_score / (2 * self.score[:, :, None])).clamp_max(1),
            (dist4 / 60).clamp_max(1),
            rival_boosting.float(),
            rival_turn.clamp(-1, 1),
            exists.float()), -1)
        obs[:, :, 32:76] = torch.where(exists[..., None], rival_block, 0.).reshape(a, n, 44)
        obs[:, :, 12] = torch.where(exists[:, :, 0], (dist4[:, :, 0] / 60).clamp_max(1), 1.)
        body_clear = torch.empty((a, n, 24), device=d)
        head_clear = torch.empty((a, n, 24), device=d)
        rays_kernel[(a, n, 24)](self.body, self.length, self.alive, h, self.length.amax(1),
                                body_clear, head_clear, n, self.s, 256)
        ray_angle = h[:, :, None] + torch.arange(24, device=d) * (2 * math.pi / 24)
        dot = head[..., 0, None] * ray_angle.cos() + head[..., 1, None] * ray_angle.sin()
        border = -dot + (dot.square() + self.radius[:, None, None].square() - norm[:, :, None].square()).clamp_min(0).sqrt()
        obs[:, :, 76::4] = body_clear
        obs[:, :, 77::4] = head_clear
        obs[:, :, 78::4] = (border / 40).clamp_max(1)
        obs[:, :, 79::4] = sector_food
        return obs

    def _push(self, single):
        if not self.stack_ready:
            self.stack.copy_(single[:, :, None].expand(-1, -1, 4, -1))
            self.stack_ready = True
        else:
            shifted = torch.empty_like(self.stack)
            shifted[:, :, :3].copy_(self.stack[:, :, 1:])
            shifted[:, :, 3].copy_(single)
            mask = self._stack_reset_mask[:, :, None, None]
            self.stack.copy_(torch.where(mask, single[:, :, None].expand(-1, -1, 4, -1), shifted))
            self._stack_reset_mask.zero_()
        flat = self.stack.reshape(self.a * self.n, 688)
        if self.personalities:
            self.last_observation[:, :688].copy_(flat)
            self.last_observation[:, 688:].copy_(self.personality_onehot)
        else:
            self.last_observation.copy_(flat)
        return self.last_observation

    def observe(self):
        if not self.stack_ready:
            return self._push(self.observe_single())
        return self.last_observation

    def scripted_actions(self, obs):
        """Device-only food/clearance controller, not the Go expert planner."""
        frame = obs[:, :688].reshape(-1, 4, 172)[:, -1]
        rays = frame[:, 76:].reshape(-1, 24, 4)
        goal_angle = torch.atan2(frame[:, 17], frame[:, 16])
        turns = self.turns[None]
        # Probe forward and the two neighboring sectors at each candidate turn.
        index = (turns / (2 * math.pi / 24)).round().long().remainder(24).expand(len(obs), -1)
        body = rays[:, :, 0].gather(1, index)
        border = rays[:, :, 2].gather(1, index)
        cost = (goal_angle[:, None] - turns).abs() + (1 - body) * 8 + (1 - border) * 6
        return cost.argmin(1)

    def _respawn(self, done):
        angle = ((self.worm[None, :, None] + .5 + (self.random(self.a, self.n, 16) - .5) * .2) / self.n) * (2 * math.pi)
        radius = (self.radius[:, None, None] - 30) * (.85 + .15 * self.random(self.a, self.n, 16))
        candidates = torch.stack((angle.cos() * radius, angle.sin() * radius), -1)
        new_heading = self.random(self.a, self.n) * (2 * math.pi)
        spawn_kernel[(self.a, self.n)](self.body, self.length, self.alive, self.heading,
            self.target, self.score, self.kills, self.mutual, self.food, self.spent,
            done, candidates, new_heading, self.length.amax(1), self.n, self.s, 16, 128,
            triton.next_power_of_2(self.s))
        ranks = done.cumsum(1, dtype=torch.int32)
        self.ids.copy_(torch.where(done, self.next_id[:, None] + ranks, self.ids))
        self.next_id.add_(done.sum(1).int())
        self.alive |= done
        self.loop_origin.copy_(torch.where(done[:, :, None], self.body[:, 0], self.loop_origin))
        self.loop_heading.copy_(torch.where(done, self.heading, self.loop_heading))
        self.loop_turn.masked_fill_(done, 0)
        self.loop_steps.masked_fill_(done, 0)
        self.prev_heading.copy_(torch.where(done, self.heading, self.prev_heading))
        self.turn_rate.masked_fill_(done, 0)
        self.boosting.masked_fill_(done, False)
        # Respawned worms start a fresh frame stack on the next push.
        self._stack_reset_mask = done.clone()

    @torch.no_grad()
    def step(self, actions, respawn=True):
        actions = torch.as_tensor(actions, device=self.device).reshape(self.a, self.n)
        if actions.shape != self.score.shape:
            raise ValueError("wrong action dimensions")
        if self.last_observation is not None:
            scripted = self.scripted_actions(self.last_observation).reshape(self.a, self.n)
            actions = torch.where(actions < 0, scripted, actions)
        old_score, old_food, old_kills, old_mutual = [x.clone() for x in (self.score, self.food, self.kills, self.mutual)]
        self.killed_size.zero_()
        self.corpse_eaten.zero_()
        self.target.copy_(self.heading + self.turns[actions.remainder(7)])
        self.boost.copy_(actions >= 7)
        for _ in range(4):
            if self.age < self.horizon:
                self._tick()
                self.age += 1
        dead = ~self.alive
        if self.personalities:
            # Personalities: 0=Aggressive, 1=Greedy, 2=Conservative, 3=Afraid
            is_agg = (self.personality_id == 0)
            # For aggressive worms: kill reward is proportional to the size of the worm killed (normalized by starter score 12),
            # plus a substantial reward proportional to how much of the killed worm's corpse it ate.
            agg_kill_reward = self.w_kill * (self.killed_size / 12.0) + 2.5 * self.corpse_eaten
            other_kill_reward = self.w_kill * (self.kills - old_kills)
            kill_reward = torch.where(is_agg, agg_kill_reward, other_kill_reward)
            reward = self.w_score * (self.score - old_score) + kill_reward - .01 - dead * self.w_death
        else:
            reward = .3 * (self.score - old_score) + 6 * (self.kills - old_kills) - .01 - dead * 10
        turned = self.heading - self.loop_heading
        self.loop_turn += (turned + math.pi).remainder(2 * math.pi) - math.pi
        self.loop_heading.copy_(self.heading)
        self.loop_steps += 1
        circles = (self.loop_turn.abs() > 5) & ((self.body[:, 0] - self.loop_origin).norm(dim=-1) < 5)
        reset_motion = circles | (self.loop_steps >= 60)
        self.loop_origin.copy_(torch.where(reset_motion[:, :, None], self.body[:, 0], self.loop_origin))
        self.loop_turn.masked_fill_(reset_motion, 0)
        self.loop_steps.masked_fill_(reset_motion, 0)
        state = {"rewards": reward.flatten(), "dones": dead.flatten().clone(),
                 "agent_food": (self.food - old_food).flatten(),
                 "agent_kills": (self.kills - old_kills).flatten(),
                 "agent_mutual_kills": (self.mutual - old_mutual).flatten(),
                 "agent_deaths": dead.flatten().clone(), "agent_circles": circles.flatten()}
        if respawn:
            self._respawn(dead)
            if self.age >= self.horizon:
                reset_state = self.reset()
                state["dones"].fill_(True)
                state.update(reset_state)
                return state
        state.update(self.state())
        if self.personalities:
            newest_dist = self.last_observation[:, 3 * 172 + 12].reshape(self.a, self.n)
            proximity_penalty = self.w_fear * (0.4 - newest_dist).clamp_min(0)
            state["rewards"] = state["rewards"] - proximity_penalty.flatten()
        return state

    def state(self, teach=False):
        current = self._push(self.observe_single())
        result = {"observations": self.last_observation, "scenarios": self.scenarios}
        if teach:
            result["teachers"] = self.scripted_actions(self.last_observation)
        return result

    def check_capacity(self):
        code = int(self.overflow.item())
        if code:
            raise RuntimeError(f"GPU simulation capacity exceeded (code {code}); increase "
                               "--max-segments / --max-pellets and resume the checkpoint")

    def close(self):
        pass

    def load_fixture(self, state):
        """Test-only controlled initialization from the authoritative Go trace."""
        snakes = sorted(state["snakes"], key=lambda snake: snake["ID"])
        if self.a != 1 or len(snakes) != self.n:
            raise ValueError("fixture dimensions do not match simulator")
        self.radius.fill_(state["radius"])
        self.tick, self.age = state["tick"], 0
        self.pellet_target.zero_()
        self.active.zero_()
        self.free_count.fill_(self.p)
        self.free_list.copy_(self.pellet_index[None])
        self.pellet_killer.fill_(-1)
        self.killed_size.zero_()
        self.corpse_eaten.zero_()
        self.tick_killer.fill_(-1)
        self.overflow.zero_()
        for i, snake in enumerate(snakes):
            segments = torch.tensor([[p["X"], p["Y"]] for p in snake["Segments"]], device=self.device)
            self.body[0, :len(segments), i] = segments
            self.length[0, i] = len(segments)
            for tensor, key in ((self.heading, "Heading"), (self.score, "Score"),
                                (self.target, "TargetHeading"), (self.alive, "Alive"),
                                (self.ids, "ID"), (self.kills, "Kills"),
                                (self.mutual, "MutualKills"), (self.food, "FoodEaten"),
                                (self.spent, "BoostSpent")):
                tensor[0, i] = snake[key]
        pellets = state["pellets"] or []
        if pellets:
            positions = torch.tensor([[[p["Pos"]["X"], p["Pos"]["Y"]] for p in pellets]], device=self.device)
            values = torch.tensor([[p["Value"] for p in pellets]], device=self.device)
            self._allocate(positions, values, torch.ones_like(values, dtype=torch.bool))
        self.next_id.fill_(max(s["ID"] for s in snakes))
        self.loop_origin.copy_(self.body[:, 0])
        self.loop_heading.copy_(self.heading)
        self.loop_turn.zero_()
        self.loop_steps.zero_()
        self.prev_heading.copy_(self.heading)
        self.turn_rate.zero_()
        self.boosting.zero_()
        self.boost.zero_()
        self.stack_ready = False
        self._stack_reset_mask.zero_()
        return self.observe()


class GPUBridge:
    """Training-side adapter with the Bridge protocol and device tensor results."""
    def __init__(self, device="cuda", max_segments=512, max_pellets=16384, graphs=False, personalities=False):
        self.graphs = graphs
        self.personalities = personalities
        self.replay = None
        self.device, self.max_segments, self.max_pellets = device, max_segments, max_pellets
        self.sim = None

    def call(self, op, **request):
        if op == "reset_batch":
            self.sim = GPUSim(request["arenas"], request["agents"], request["horizon"],
                              request["seed"], self.device, self.max_segments,
                              self.max_pellets, request.get("curriculum", False),
                              request.get("stage", 0), self.personalities)
            self.replay = None
            state = {"observations": self.sim.last_observation,
                     "scenarios": self.sim.scenarios}
        elif op == "step_batch":
            if self.graphs:
                if self.replay is None:
                    self.replay = GraphStep(self.sim)
                state = self.replay.step(request["actions"])
            else:
                state = self.sim.step(request["actions"])
        elif op == "curriculum":
            self.sim.stage = request["stage"]
            return {"ok": True}
        else:
            raise ValueError(f"unsupported GPU training operation {op}")
        if request.get("teach"):
            state["teachers"] = self.sim.scripted_actions(state["observations"])
        return state

    def check_capacity(self):
        if self.sim is not None:
            self.sim.check_capacity()

    def close(self):
        self.sim = None


class GraphStep:
    """Replay one four-tick transition without Python dispatch between kernels.

    Warm-up/capture state and RNG changes are restored. Arena resets are handled
    outside the graph because they change profile metadata and tensor storage.
    """
    def __init__(self, sim):
        self.sim = sim
        self.actions = torch.full((sim.a * sim.n,), 3, device=sim.device, dtype=torch.long)
        self.graph = torch.cuda.CUDAGraph()
        self.invalidated = False
        self._capture()

    def _capture(self):
        sim = self.sim
        tensors = {name: tensor.clone() for name, tensor in vars(sim).items()
                   if isinstance(tensor, torch.Tensor)}
        age, tick, horizon = sim.age, sim.tick, sim.horizon
        rng = sim.generator.get_state()
        sim.horizon = 1_000_000_000
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                sim.step(self.actions)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Every mutable tensor retains its address across decisions. Restore
        # values rather than rebinding pointers captured by the graph.
        for name, value in tensors.items():
            getattr(sim, name).copy_(value)
        sim.age, sim.tick = age, tick
        sim.generator.set_state(rng)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = sim.step(self.actions)
        torch.cuda.current_stream().wait_stream(stream)
        for name, value in tensors.items():
            getattr(sim, name).copy_(value)
        sim.generator.set_state(rng)
        sim.age, sim.tick, sim.horizon = age, tick, horizon

    def step(self, actions):
        sim = self.sim
        if sim.horizon - sim.age < 4:
            result=sim.step(actions)
            self.invalidated=True
            return result
        if self.invalidated:
            self.graph = torch.cuda.CUDAGraph()
            self._capture()
            self.invalidated = False
        self.actions.copy_(actions.reshape(-1))
        self.graph.replay()
        sim.age += 4
        sim.tick += 4
        result = dict(self.output)
        result["scenarios"] = sim.scenarios
        if sim.age >= sim.horizon:
            sim.reset()
            result["dones"].fill_(True)
            result["observations"] = sim.last_observation
            result["scenarios"] = sim.scenarios
            self.invalidated = True
        return result
