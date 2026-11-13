//! Transport-free, deterministic fixed-step snake arena. No wall clock or threads.
use std::collections::{BTreeMap, VecDeque};
use std::f32::consts::{PI, TAU};

pub const RADIUS: f32 = 600.0;
pub const TOTAL_MASS: u32 = 20_000;
pub const START_MASS: u32 = 100;
pub const MAX_SNAKES: usize = 24;
pub const MAX_FOOD: usize = 8192;
pub const DT: f32 = 1.0 / 30.0;
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}
impl Point {
    pub fn distance(self, b: Self) -> f32 {
        ((self.x - b.x).powi(2) + (self.y - b.y).powi(2)).sqrt()
    }
    pub fn norm(self) -> f32 {
        self.distance(Self::default())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Action {
    pub heading: f32,
    pub boost: bool,
}
#[derive(Clone, Debug)]
pub struct Snake {
    pub id: u64,
    pub name: String,
    pub color: usize,
    pub mass: u32,
    pub heading: f32,
    pub action: Action,
    pub body: VecDeque<Point>,
    pub age: u64,
    pub crown_at: Option<u64>,
    pub boosting: bool,
}
impl Snake {
    pub fn head(&self) -> Point {
        self.body[0]
    }
    pub fn width(&self) -> f32 {
        (2.5 * (self.mass as f32 / 100.0).powf(0.22)).clamp(2.5, 7.0)
    }
    pub fn length(&self) -> f32 {
        20.0 + 0.08 * self.mass.saturating_sub(100) as f32
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Food {
    pub pos: Point,
    pub mass: u32,
    pub color: usize,
    pub expires: u64,
}
#[derive(Clone, Debug, PartialEq)]
pub enum DeathReason {
    Boundary,
    Collision,
    CrownExpired,
}
#[derive(Clone, Debug)]
pub struct Death {
    pub id: u64,
    pub reason: DeathReason,
}
#[derive(Clone, Debug)]
pub struct World {
    pub tick: u64,
    pub snakes: BTreeMap<u64, Snake>,
    pub food: Vec<Food>,
    pub reserve: u32,
    pub deaths: Vec<Death>,
    rng: u64,
    next_id: u64,
}
fn angle_delta(a: f32, b: f32) -> f32 {
    (a - b + PI).rem_euclid(TAU) - PI
}
fn segment_distance(p: Point, a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let t =
        (((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy).max(1e-12)).clamp(0.0, 1.0);
    p.distance(Point {
        x: a.x + t * dx,
        y: a.y + t * dy,
    })
}
fn swept_distance(a: Point, b: Point, c: Point, d: Point) -> f32 {
    fn cross(a: Point, b: Point, c: Point) -> f32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
    if cross(a, b, c) * cross(a, b, d) < 0.0 && cross(c, d, a) * cross(c, d, b) < 0.0 {
        return 0.0;
    }
    segment_distance(a, c, d)
        .min(segment_distance(b, c, d))
        .min(segment_distance(c, a, b))
        .min(segment_distance(d, a, b))
}
impl World {
    pub fn new(seed: u64) -> Self {
        let mut w = Self {
            tick: 0,
            snakes: BTreeMap::new(),
            food: Vec::new(),
            reserve: TOTAL_MASS,
            deaths: Vec::new(),
            rng: seed.max(1),
            next_id: 1,
        };
        for _ in 0..2000 {
            w.ambient();
        }
        w
    }
    fn random(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u32 << 24) as f32
    }
    fn random_point(&mut self) -> Point {
        let r = self.random().sqrt() * (RADIUS - 20.0);
        let a = self.random() * TAU;
        Point {
            x: r * a.cos(),
            y: r * a.sin(),
        }
    }
    fn ambient(&mut self) {
        let mass = [1, 4, 12][(self.random() * 3.0) as usize];
        if self.reserve < mass || self.food.len() >= MAX_FOOD {
            return;
        }
        let pos = self.random_point();
        // No free food placed inside a body; keep the reserve for another attempt.
        if self
            .snakes
            .values()
            .any(|s| s.body.iter().any(|p| p.distance(pos) < s.width() + 12.0))
        {
            return;
        }
        let color = (self.random() * 6.0) as usize;
        self.reserve -= mass;
        self.food.push(Food {
            pos,
            mass,
            color,
            expires: self.tick + 1800,
        });
    }
    pub fn spawn(&mut self, name: &str, color: usize) -> Option<u64> {
        if self.snakes.len() >= MAX_SNAKES || self.reserve < START_MASS {
            return None;
        }
        let mut pos = None;
        for _ in 0..64 {
            let p = self.random_point();
            if self
                .snakes
                .values()
                .all(|s| s.body.iter().all(|q| q.distance(p) > 65.0))
            {
                pos = Some(p);
                break;
            }
        }
        let pos = pos?;
        let heading = self.random() * TAU;
        let id = self.next_id;
        self.next_id += 1;
        self.reserve -= START_MASS;
        // New bodies unfurl over the first few ticks; no invulnerable spawn shield.
        self.snakes.insert(
            id,
            Snake {
                id,
                name: name
                    .chars()
                    .filter(|c| c.is_ascii_graphic() || *c == ' ')
                    .take(16)
                    .collect(),
                color: color % 6,
                mass: START_MASS,
                heading,
                action: Action {
                    heading,
                    boost: false,
                },
                body: VecDeque::from([pos]),
                age: 0,
                crown_at: None,
                boosting: false,
            },
        );
        Some(id)
    }
    pub fn action(&mut self, id: u64, action: Action) {
        if action.heading.is_finite()
            && let Some(s) = self.snakes.get_mut(&id)
        {
            s.action = Action {
                heading: action.heading.rem_euclid(TAU),
                ..action
            };
        }
    }
    pub fn remove(&mut self, id: u64) {
        if let Some(s) = self.snakes.remove(&id) {
            self.reserve += s.mass;
        }
    }
    pub fn total_mass(&self) -> u32 {
        self.reserve
            + self.food.iter().map(|f| f.mass).sum::<u32>()
            + self.snakes.values().map(|s| s.mass).sum::<u32>()
    }
    pub fn step(&mut self) {
        self.tick += 1;
        self.deaths.clear();
        let old: BTreeMap<_, _> = self.snakes.iter().map(|(&id, s)| (id, s.head())).collect();
        for s in self.snakes.values_mut() {
            s.age += 1;
            if s.crown_at.is_none() && (s.mass >= 1800 || s.age >= 11700) {
                s.crown_at = Some(self.tick);
            }
            s.boosting = s.action.boost && s.mass > START_MASS;
            let turn = (2.8 / (s.mass as f32 / 100.0).powf(0.12)) * DT;
            s.heading += angle_delta(s.action.heading, s.heading).clamp(-turn, turn);
            let speed =
                38.0 / (s.mass as f32 / 100.0).powf(0.04) * if s.boosting { 1.65 } else { 1.0 };
            let h = s.head();
            s.body.push_front(Point {
                x: h.x + s.heading.cos() * speed * DT,
                y: h.y + s.heading.sin() * speed * DT,
            });
            let mut length = 0.0;
            let mut keep = 1;
            for i in 1..s.body.len() {
                length += s.body[i - 1].distance(s.body[i]);
                keep = i + 1;
                if length >= s.length() {
                    break;
                }
            }
            s.body.truncate(keep.min(256));
            if self.tick.is_multiple_of(10) && s.boosting {
                s.mass -= 1;
                self.reserve += 1;
            }
            if self.tick.is_multiple_of(30) && s.mass > 1200 {
                let cost = (s.mass - 1200) / 200 + 1;
                s.mass -= cost;
                self.reserve += cost;
            }
        }
        let mut dead = BTreeMap::new();
        for (&id, s) in &self.snakes {
            if s.head().norm() + s.width() >= RADIUS {
                dead.insert(id, DeathReason::Boundary);
            }
            if s.crown_at.is_some_and(|t| self.tick - t >= 2700) {
                dead.insert(id, DeathReason::CrownExpired);
            }
            for (&other, b) in &self.snakes {
                if id == other {
                    continue;
                }
                if swept_distance(old[&id], s.head(), old[&other], b.head()) < s.width() + b.width()
                {
                    dead.insert(id, DeathReason::Collision);
                    dead.insert(other, DeathReason::Collision);
                }
                for i in 1..b.body.len() {
                    if swept_distance(old[&id], s.head(), b.body[i - 1], b.body[i])
                        < s.width() + b.width()
                    {
                        dead.insert(id, DeathReason::Collision);
                        break;
                    }
                }
            }
        }
        // Resolve all deaths before pickup; no iteration-order survivor advantage.
        for (id, reason) in dead {
            let s = self.snakes.remove(&id).unwrap();
            let loot = s.mass * 40 / 100;
            self.reserve += s.mass - loot;
            if self.food.len() < MAX_FOOD {
                self.food.push(Food {
                    pos: s.head(),
                    mass: loot,
                    color: s.color,
                    expires: self.tick + 600,
                });
            } else {
                self.reserve += loot;
            }
            self.deaths.push(Death { id, reason });
        }
        let mut remaining = Vec::with_capacity(self.food.len());
        for f in self.food.drain(..) {
            if f.expires <= self.tick {
                self.reserve += f.mass;
                continue;
            }
            // Closest eligible head gets contested food; stable ID only breaks exact ties.
            let winner = self
                .snakes
                .values()
                .filter(|s| {
                    s.mass < 3000 && segment_distance(f.pos, old[&s.id], s.head()) < s.width() + 2.0
                })
                .min_by(|a, b| {
                    a.head()
                        .distance(f.pos)
                        .total_cmp(&b.head().distance(f.pos))
                        .then(a.id.cmp(&b.id))
                })
                .map(|s| s.id);
            if let Some(id) = winner {
                let s = self.snakes.get_mut(&id).unwrap();
                let gained = f.mass.min(3000 - s.mass);
                s.mass += gained;
                self.reserve += f.mass - gained;
            } else {
                remaining.push(f);
            }
        }
        self.food = remaining;
        if self.tick.is_multiple_of(3) && self.food.iter().map(|f| f.mass).sum::<u32>() < 8000 {
            for _ in 0..4 {
                self.ambient();
            }
        }
        debug_assert_eq!(self.total_mass(), TOTAL_MASS);
    }
    /// Local observation baseline: food and bodies outside this radius are ignored.
    pub fn heuristic(&self, id: u64) -> Action {
        let Some(s) = self.snakes.get(&id) else {
            return Action::default();
        };
        let h = s.head();
        let mut best = (f32::NEG_INFINITY, s.heading);
        for delta in [-1.2, -0.6, 0.0, 0.6, 1.2] {
            let a = s.heading + delta;
            let p = Point {
                x: h.x + a.cos() * 45.0,
                y: h.y + a.sin() * 45.0,
            };
            let mut score = -delta.abs() * 0.2;
            if p.norm() > RADIUS - 25.0 {
                score -= 1000.0;
            }
            for b in self.snakes.values().filter(|b| b.id != id) {
                for q in b.body.iter().filter(|q| q.distance(h) < 150.0) {
                    score -= 150.0 / (p.distance(*q) - s.width() - b.width()).max(0.1);
                }
            }
            for f in self.food.iter().filter(|f| f.pos.distance(h) < 150.0) {
                score += f.mass.min(12) as f32 / (p.distance(f.pos) + 5.0);
            }
            if score > best.0 {
                best = (score, a);
            }
        }
        Action {
            heading: best.1,
            boost: false,
        }
    }
    /// Regression fingerprint for this rules/build/target; not a cross-platform float promise.
    pub fn state_hash(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        let mut add = |v: u64| {
            h = (h ^ v).wrapping_mul(0x100000001b3);
        };
        add(self.tick);
        add(self.reserve as u64);
        add(self.rng);
        add(self.next_id);
        for s in self.snakes.values() {
            add(s.id);
            add(s.mass as u64);
            add(s.heading.to_bits() as u64);
            add(s.action.heading.to_bits() as u64);
            add(s.action.boost as u64);
            add(s.age);
            add(s.crown_at.unwrap_or(0));
            for p in &s.body {
                add(p.x.to_bits() as u64);
                add(p.y.to_bits() as u64);
            }
        }
        for f in &self.food {
            add(f.pos.x.to_bits() as u64);
            add(f.pos.y.to_bits() as u64);
            add(f.mass as u64);
            add(f.expires);
        }
        h
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_and_conserved() {
        let mut a = World::new(42);
        let mut b = a.clone();
        for i in 0..8 {
            a.spawn("bot", i);
            b.spawn("bot", i);
        }
        for tick in 0..900 {
            for w in [&mut a, &mut b] {
                let actions: Vec<_> = w.snakes.keys().map(|&id| (id, w.heuristic(id))).collect();
                for (id, act) in actions {
                    w.action(id, act);
                }
                w.step();
                if tick % 90 == 0 {
                    w.spawn("new", 0);
                }
            }
            assert_eq!(a.total_mass(), TOTAL_MASS);
            assert_eq!(a.state_hash(), b.state_hash());
            assert!(a.food.len() <= MAX_FOOD);
        }
    }
    #[test]
    fn crossing_heads_die_together() {
        let mut w = World::new(1);
        w.food.clear();
        w.reserve = TOTAL_MASS;
        let a = w.spawn("a", 0).unwrap();
        let b = w.spawn("b", 1).unwrap();
        for (id, x, heading) in [(a, -2.6, 0.0), (b, 2.6, PI)] {
            let s = w.snakes.get_mut(&id).unwrap();
            s.body = VecDeque::from([Point { x, y: 0.0 }]);
            s.heading = heading;
            s.action.heading = heading;
        }
        w.step();
        assert!(w.snakes.is_empty());
        assert_eq!(w.total_mass(), TOTAL_MASS);
        assert_eq!(w.deaths.len(), 2);
    }
    #[test]
    fn crown_persists_after_shedding() {
        let mut w = World::new(7);
        let id = w.spawn("champ", 0).unwrap();
        let s = w.snakes.get_mut(&id).unwrap();
        s.crown_at = Some(1);
        s.age = 0;
        w.tick = 2700;
        w.step();
        assert!(!w.snakes.contains_key(&id));
        assert_eq!(w.total_mass(), TOTAL_MASS);
    }
    #[test]
    fn invalid_actions_and_capacity() {
        let mut w = World::new(12);
        let id = w.spawn("safe\x1b[31m", 0).unwrap();
        let h = w.snakes[&id].action.heading;
        w.action(
            id,
            Action {
                heading: f32::NAN,
                boost: true,
            },
        );
        assert_eq!(w.snakes[&id].action.heading, h);
        for _ in 0..100 {
            w.spawn("bot", 0);
        }
        assert!(w.snakes.len() <= MAX_SNAKES);
        assert_eq!(w.total_mass(), TOTAL_MASS);
        let ids: Vec<_> = w.snakes.keys().copied().collect();
        for id in ids {
            w.remove(id);
        }
        assert_eq!(w.total_mass(), TOTAL_MASS);
    }
}
