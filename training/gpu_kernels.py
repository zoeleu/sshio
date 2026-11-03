"""CUDA and ROCm/Triton kernels for the sequential body constraint and arena reductions.

All pointers refer to device tensors. Kernels never call the Go simulator or
copy observations/actions to the host. Bodies use [arena, segment, worm, xy].
"""
import triton as tr
import triton.language as tl

try:
    from triton.language.extra.hip import libdevice as _hip_libdevice
except ImportError:
    _hip_libdevice = None

try:
    from triton.language.extra.cuda import libdevice as _cuda_libdevice
except ImportError:
    _cuda_libdevice = None

libdevice = _hip_libdevice if (_hip_libdevice is not None and hasattr(_hip_libdevice, "atan2")) else _cuda_libdevice
if libdevice is None:
    from triton.language.extra import libdevice



@tr.jit
def move_kernel(body, heading, target, score, length, alive, boost, spent,
                drop_pos, drop_mask, N: tl.constexpr, S: tl.constexpr,
                FIRST: tl.constexpr, DROP: tl.constexpr, BLOCK: tl.constexpr):
    arena = tl.program_id(0)
    worm = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = worm < N
    ix = arena * N + worm
    living = tl.load(alive + ix, valid, other=0)
    active = valid & living
    h = tl.load(heading + ix, valid, other=0)
    t = tl.load(target + ix, valid, other=0)
    delta = t - h
    delta -= tl.floor((delta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
    h += tl.minimum(0.045, tl.maximum(-0.045, delta))
    sc = tl.load(score + ix, valid, other=12)
    fast = (sc > 12) & tl.load(boost + ix, valid, other=0) & active
    size = tl.load(length + ix, valid, other=1)
    if FIRST:
        tail = ((arena * S + size - 1) * N + worm) * 2
        tl.store(drop_pos + ix * 2, tl.load(body + tail, active, other=0), valid)
        tl.store(drop_pos + ix * 2 + 1, tl.load(body + tail + 1, active, other=0), valid)
        tl.store(drop_mask + ix, fast & DROP, valid)
        cost = tl.where(fast, tl.minimum(0.2, sc - 12), 0)
        tl.store(score + ix, sc - cost, active)
        tl.store(spent + ix, tl.load(spent + ix, valid, other=0) + cost, active)
    offset = (arena * S * N + worm) * 2
    distance = tl.where(fast, 0.4, 0.2)
    x = tl.load(body + offset, valid, other=0) + tl.cos(h) * distance
    y = tl.load(body + offset + 1, valid, other=0) + tl.sin(h) * distance
    tl.store(body + offset, x, active)
    tl.store(body + offset + 1, y, active)
    tl.store(heading + ix, h, active)
    # Preserve Go's ordered follow constraint, rather than approximating the
    # body with a delayed head trail or a parallel Jacobi relaxation.
    maximum = tl.minimum(tl.max(tl.where(active, size, 1), 0), S)
    for segment in range(1, maximum):
        p = ((arena * S + segment) * N + worm) * 2
        mask = active & (segment < size)
        bx = tl.load(body + p, mask, other=0)
        by = tl.load(body + p + 1, mask, other=0)
        dx, dy = x - bx, y - by
        d = tl.sqrt(dx * dx + dy * dy)
        follow = d > 0.65
        nx = tl.where(follow, x - dx * (0.65 / tl.maximum(d, 1.e-8)), bx)
        ny = tl.where(follow, y - dy * (0.65 / tl.maximum(d, 1.e-8)), by)
        tl.store(body + p, nx, mask)
        tl.store(body + p + 1, ny, mask)
        x, y = nx, ny


@tr.jit
def collision_kernel(body, length, score, alive, ids, radius, maximum,
                     dead, killer, N: tl.constexpr, S: tl.constexpr,
                     BLOCK: tl.constexpr):
    arena, worm = tl.program_id(0), tl.program_id(1)
    ix = arena * N + worm
    living = tl.load(alive + ix)
    x = tl.load(body + (arena * S * N + worm) * 2)
    y = tl.load(body + (arena * S * N + worm) * 2 + 1)
    my_score = tl.load(score + ix)
    my_scale = 1.0 + 3.0 * tl.minimum(1.0, tl.maximum(0.0, (my_score - 12.0) / 180.0))
    best_id, owner = tl.full((), 2147483647, tl.int32), tl.full((), -1, tl.int32)
    offsets = tl.arange(0, BLOCK)
    if living:
        limit = tl.minimum(tl.load(maximum + arena), S) * N
        for start in range(0, limit, BLOCK):
            flat = start + offsets
            other, segment = flat % N, flat // N
            valid = (flat < limit) & (other != worm)
            valid &= tl.load(alive + arena * N + other, flat < limit, other=0)
            valid &= segment < tl.load(length + arena * N + other, flat < limit, other=0)
            other_score = tl.load(score + arena * N + other, flat < limit, other=12.0)
            other_scale = 1.0 + 3.0 * tl.minimum(1.0, tl.maximum(0.0, (other_score - 12.0) / 180.0))
            hit_r = 0.425 * (my_scale + other_scale)
            p = (arena * S * N + flat) * 2
            dx = tl.load(body + p, valid, other=1.e10) - x
            dy = tl.load(body + p + 1, valid, other=1.e10) - y
            hit = valid & (dx * dx + dy * dy < hit_r * hit_r)
            candidate = tl.where(hit, tl.load(ids + arena * N + other, flat < limit, other=2147483647), 2147483647)
            local_id = tl.min(candidate, 0)
            local_owner = tl.min(tl.where(candidate == local_id, other, 2147483647), 0)
            owner = tl.where(local_id < best_id, local_owner, owner)
            best_id = tl.minimum(best_id, local_id)
    r = tl.load(radius + arena)
    dies = living & ((x * x + y * y > r * r) | (owner >= 0))
    tl.store(dead + ix, dies)
    tl.store(killer + ix, tl.where(living, owner, -1))


@tr.jit
def resolve_kernel(alive, dead, killer, score, boosting, kills, mutual, tick_dead,
                   killed_size, tick_killer,
                   TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = ix < TOTAL
    dies = tl.load(dead + ix, valid, other=0)
    owner = tl.load(killer + ix, valid, other=-1)
    arena_base = (ix // N) * N
    target = arena_base + owner
    has_owner = valid & dies & (owner >= 0)
    target_dies = tl.load(dead + target, has_owner, other=0)
    target_owner = tl.load(killer + target, has_owner, other=-1)
    worm_idx = ix % N
    is_mutual = has_owner & target_dies & (target_owner == worm_idx)
    my_boost = tl.load(boosting + ix, is_mutual, other=0)
    other_boost = tl.load(boosting + target, is_mutual, other=0)
    my_score = tl.load(score + ix, is_mutual, other=12.0)
    other_score = tl.load(score + target, is_mutual, other=12.0)
    i_win = is_mutual & ((my_boost & ~other_boost) | ((my_boost == other_boost) & (my_score > other_score)))
    other_wins = is_mutual & ((other_boost & ~my_boost) | ((my_boost == other_boost) & (other_score > my_score)))
    actual_dies = dies & ~i_win
    credited = valid & actual_dies & (owner >= 0)
    both_die = is_mutual & ~i_win & ~other_wins
    tl.atomic_add(kills + target, 1, credited)
    tl.atomic_add(mutual + target, 1, credited & both_die)
    victim_score = tl.load(score + ix, credited, other=0.0)
    tl.atomic_add(killed_size + target, victim_score, credited)
    tl.store(tick_killer + ix, owner, credited)
    tl.store(alive + ix, tl.load(alive + ix, valid, other=0) & ~actual_dies, valid)
    tl.store(tick_dead + ix, tl.load(tick_dead + ix, valid, other=0) | actual_dies, valid)


@tr.jit
def resize_kernel(body, score, length, alive, overflow,
                  N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    arena = tl.program_id(0)
    worm = tl.arange(0, BLOCK)
    valid = worm < N
    ix = arena * N + worm
    living = tl.load(alive + ix, valid, other=0)
    old = tl.load(length + ix, valid, other=1)
    new = tl.load(score + ix, valid, other=12).to(tl.int32) + 8
    tl.atomic_max(overflow, tl.max(tl.where(valid & (new > S), 1, 0), 0))
    new = tl.minimum(new, S)
    tail = ((arena * S + old - 1) * N + worm) * 2
    x = tl.load(body + tail, valid, other=0)
    y = tl.load(body + tail + 1, valid, other=0)
    minimum = tl.min(tl.where(valid & living, old, S), 0)
    maximum = tl.max(tl.where(valid & living, new, 1), 0)
    for segment in range(minimum, maximum):
        p = ((arena * S + segment) * N + worm) * 2
        grow = valid & living & (segment >= old) & (segment < new)
        tl.store(body + p, x, grow)
        tl.store(body + p + 1, y, grow)
    tl.store(length + ix, new, valid & living)


@tr.jit
def allocate_kernel(free_list, free_count, ranks, request_mask, request_pos,
                    request_value, request_killer, pos, value, pellet_killer, active, overflow,
                    P: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    arena = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = i < R
    requested = tl.load(request_mask + arena * R + i, valid, other=0)
    rank = tl.load(ranks + arena * R + i, valid, other=0)
    available = tl.load(free_count + arena)
    fit = valid & requested & (rank <= available)
    slot = tl.load(free_list + arena * P + available - rank, fit, other=0)
    out = arena * P + slot
    source = arena * R + i
    tl.store(pos + out * 2, tl.load(request_pos + source * 2, fit, other=0), fit)
    tl.store(pos + out * 2 + 1, tl.load(request_pos + source * 2 + 1, fit, other=0), fit)
    tl.store(value + out, tl.load(request_value + source, fit, other=0), fit)
    tl.store(pellet_killer + out, tl.load(request_killer + source, fit, other=-1), fit)
    tl.store(active + out, 1, fit)
    exceeded = tl.sum((valid & requested & ~fit).to(tl.int32), 0) > 0
    tl.atomic_max(overflow, tl.where(exceeded, 2, 0))


@tr.jit
def release_kernel(free_list, free_count, released, ranks, active,
                   P: tl.constexpr, BLOCK: tl.constexpr):
    arena = tl.program_id(0)
    slot = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = slot < P
    mask = valid & tl.load(released + arena * P + slot, valid, other=0)
    rank = tl.load(ranks + arena * P + slot, valid, other=0)
    available = tl.load(free_count + arena)
    tl.store(free_list + arena * P + available + rank - 1, slot, mask)
    tl.store(active + arena * P + slot, 0, mask)


@tr.jit
def pickup_kernel(body, score, alive, ids, pos, active, owner,
                  N: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
                  WORMS: tl.constexpr, BLOCK: tl.constexpr):
    arena = tl.program_id(0)
    pellet = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    worm = tl.arange(0, WORMS)
    valid = pellet < P
    enabled = tl.load(active + arena * P + pellet, valid, other=0)
    px = tl.load(pos + (arena * P + pellet) * 2, valid, other=0)
    py = tl.load(pos + (arena * P + pellet) * 2 + 1, valid, other=0)
    hx = tl.load(body + (arena * S * N + worm) * 2, worm < N, other=1.e10)
    hy = tl.load(body + (arena * S * N + worm) * 2 + 1, worm < N, other=1.e10)
    living = tl.load(alive + arena * N + worm, worm < N, other=0)
    worm_score = tl.load(score + arena * N + worm, worm < N, other=12.0)
    worm_scale = 1.0 + 3.0 * tl.minimum(1.0, tl.maximum(0.0, (worm_score - 12.0) / 180.0))
    pickup_r = 1.3 + 0.45 * (worm_scale - 1.0)
    dx, dy = px[:, None] - hx[None, :], py[:, None] - hy[None, :]
    near = (dx * dx + dy * dy < (pickup_r * pickup_r)[None, :]) & living[None, :] & enabled[:, None]
    names = tl.load(ids + arena * N + worm, worm < N, other=2147483647)
    best = tl.min(tl.where(near, names[None, :], 2147483647), 1)
    winners = tl.min(tl.where(near & (names[None, :] == best[:, None]), worm[None, :], 2147483647), 1)
    tl.store(owner + arena * P + pellet, tl.where(best < 2147483647, winners, -1), valid)


@tr.jit
def rays_kernel(body, length, alive, heading, maximum, output_body, output_head,
                N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    arena, worm, ray = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = tl.load(heading + arena * N + worm) + ray * (6.283185307179586 / 24)
    vx, vy = tl.cos(h), tl.sin(h)
    x = tl.load(body + (arena * S * N + worm) * 2)
    y = tl.load(body + (arena * S * N + worm) * 2 + 1)
    offsets = tl.arange(0, BLOCK)
    nearest_body = tl.full((), 30., tl.float32)
    nearest_head = tl.full((), 30., tl.float32)
    limit = tl.minimum(tl.load(maximum + arena), S) * N
    for start in range(0, limit, BLOCK):
        flat = start + offsets
        other, segment = flat % N, flat // N
        valid = (flat < limit) & (other != worm)
        valid &= tl.load(alive + arena * N + other, flat < limit, other=0)
        valid &= segment < tl.load(length + arena * N + other, flat < limit, other=0)
        p = (arena * S * N + flat) * 2
        dx = tl.load(body + p, valid, other=1.e10) - x
        dy = tl.load(body + p + 1, valid, other=1.e10) - y
        along, across = dx * vx + dy * vy, tl.abs(dx * vy - dy * vx)
        hit = valid & (along >= 0) & (across < 1.6)
        dist = tl.where(hit, tl.maximum(0., along - 0.85), 30.)
        nearest_body = tl.minimum(nearest_body, tl.min(tl.where(segment == 0, 30., dist), 0))
        nearest_head = tl.minimum(nearest_head, tl.min(tl.where(segment == 0, dist, 30.), 0))
    tl.store(output_body + (arena * N + worm) * 24 + ray, nearest_body / 30.)
    tl.store(output_head + (arena * N + worm) * 24 + ray, nearest_head / 30.)


@tr.jit
def spawn_kernel(body, length, alive, heading, target, score, kills, mutual,
                 food, spent, done, candidates, new_heading, maximum,
                 N: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                 BLOCK: tl.constexpr, SEGMENTS: tl.constexpr):
    arena, worm = tl.program_id(0), tl.program_id(1)
    ix = arena * N + worm
    if tl.load(done + ix):
        candidate = tl.arange(0, K)
        x = tl.load(candidates + ((arena * N + worm) * K + candidate) * 2)
        y = tl.load(candidates + ((arena * N + worm) * K + candidate) * 2 + 1)
        nearest = tl.full((K,), 1.e20, tl.float32)
        offsets = tl.arange(0, BLOCK)
        limit = tl.minimum(tl.load(maximum + arena), S) * N
        for start in range(0, limit, BLOCK):
            flat = start + offsets
            other, segment = flat % N, flat // N
            valid = flat < limit
            valid &= tl.load(alive + arena * N + other, flat < limit, other=0)
            valid &= segment < tl.load(length + arena * N + other, flat < limit, other=0)
            p = (arena * S * N + flat) * 2
            bx = tl.load(body + p, valid, other=1.e10)
            by = tl.load(body + p + 1, valid, other=1.e10)
            dx, dy = x[:, None] - bx[None, :], y[:, None] - by[None, :]
            nearest = tl.minimum(nearest, tl.min(dx * dx + dy * dy, 1))
        first = tl.min(tl.where(nearest >= 400, candidate, K), 0)
        best = tl.argmax(nearest, 0)
        chosen = tl.where(first < K, first, best)
        px = tl.sum(tl.where(candidate == chosen, x, 0.), 0)
        py = tl.sum(tl.where(candidate == chosen, y, 0.), 0)
        h = tl.load(new_heading + ix)
        segments = tl.arange(0, SEGMENTS)
        p = ((arena * S + segments) * N + worm) * 2
        tl.store(body + p, px - tl.cos(h) * segments * 0.65, segments < 20)
        tl.store(body + p + 1, py - tl.sin(h) * segments * 0.65, segments < 20)
        tl.store(length + ix, 20)
        tl.store(score + ix, 12.)
        tl.store(heading + ix, h)
        tl.store(target + ix, h)
        tl.store(kills + ix, 0)
        tl.store(mutual + ix, 0)
        tl.store(food + ix, 0.)
        tl.store(spent + ix, 0.)


@tr.jit
def food_kernel(body, heading, radius, pos, value, active, richness, utility_out,
                sector_out, N: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
                BLOCK: tl.constexpr):
    # Writes per-pellet utility (for host-side top-4 target selection) and
    # the 24-sector food mass for one worm.
    arena, worm = tl.program_id(0), tl.program_id(1)
    ix = arena * N + worm
    x = tl.load(body + (arena * S * N + worm) * 2)
    y = tl.load(body + (arena * S * N + worm) * 2 + 1)
    heading_value = tl.load(heading + ix)
    r = tl.load(radius + arena)
    offsets, rays = tl.arange(0, BLOCK), tl.arange(0, 32)
    mass = tl.full((32,), 0., tl.float32)
    for start in range(0, P, BLOCK):
        slot = start + offsets
        enabled = (slot < P) & tl.load(active + arena * P + slot, slot < P, other=0)
        if tl.sum(enabled.to(tl.int32), 0) > 0:
            px = tl.load(pos + (arena * P + slot) * 2, enabled, other=0)
            py = tl.load(pos + (arena * P + slot) * 2 + 1, enabled, other=0)
            val = tl.load(value + arena * P + slot, enabled, other=0)
            rich = tl.load(richness + arena * P + slot, enabled, other=0)
            dx, dy = px - x, py - y
            distance = tl.sqrt(dx * dx + dy * dy)
            raw_angle = libdevice.atan2(dy, dx) - heading_value
            angle = raw_angle
            angle -= tl.floor((angle + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
            reachable = enabled & (px * px + py * py <= (r - 5) * (r - 5))
            reachable &= ~((distance < 9) & (tl.abs(angle) > 1.570796326794897))
            utility = tl.where(reachable, (val + tl.minimum(15., rich) * .3) / (8 + distance + 12 * tl.abs(angle)), -float("inf"))
            tl.store(utility_out + ix * P + slot, utility, slot < P)
            scaled = raw_angle / (6.283185307179586 / 24)
            sector = (tl.where(scaled < 0, -1., 1.) * tl.floor(tl.abs(scaled) + .5)).to(tl.int32)
            sector = (sector % 24 + 24) % 24
            weight = tl.where(enabled & (distance <= 40), val / (1 + distance) * .3, 0.)
            mass += tl.sum(tl.where(sector[None, :] == rays[:, None], weight[None, :], 0.), 1)
        else:
            tl.store(utility_out + ix * P + slot, -float("inf"), slot < P)
    tl.store(sector_out + ix * 24 + rays, tl.minimum(1., mass), rays < 24)
