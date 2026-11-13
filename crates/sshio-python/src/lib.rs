//! Bounded, single-threaded Python adapter around the canonical Rust world.
use pyo3::{exceptions::PyValueError, prelude::*};
use sshio_core::{Action, Point, RADIUS, World};
use std::f32::consts::TAU;
const OBS: usize = 76;
type TransitionBatch = (Vec<Vec<f32>>, Vec<f32>, Vec<bool>);
#[pyclass]
struct Batch {
    worlds: Vec<World>,
    agents: Vec<Vec<u64>>,
    slots: usize,
}
fn observation(w: &World, id: u64) -> Vec<f32> {
    let mut out = vec![0.0; OBS];
    let Some(s) = w.snakes.get(&id) else {
        return out;
    };
    let head = s.head();
    // 24 egocentric sectors: nearest food, nearest body, boundary clearance.
    for ray in 0..24 {
        let angle = s.heading + TAU * ray as f32 / 24.0;
        let direction = Point {
            x: angle.cos(),
            y: angle.sin(),
        };
        let sector = |p: Point| {
            let a = ((p.y - head.y).atan2(p.x - head.x) - s.heading).rem_euclid(TAU);
            ((a / TAU * 24.0 + 0.5) as usize) % 24
        };
        let mut fd = 150.0f32;
        let mut bd = 150.0f32;
        for f in &w.food {
            if head.distance(f.pos) <= 150.0 && sector(f.pos) == ray {
                fd = fd.min(head.distance(f.pos));
            }
        }
        for b in w.snakes.values().filter(|b| b.id != id) {
            for p in &b.body {
                if head.distance(*p) <= 150.0 && sector(*p) == ray {
                    bd = bd.min((head.distance(*p) - b.width() - s.width()).max(0.0));
                }
            }
        }
        let dot = head.x * direction.x + head.y * direction.y;
        let wall = -dot
            + (dot * dot + RADIUS * RADIUS - head.norm().powi(2))
                .max(0.0)
                .sqrt();
        out[ray * 3] = 1.0 - fd / 150.0;
        out[ray * 3 + 1] = 1.0 - bd / 150.0;
        out[ray * 3 + 2] = 1.0 - wall.clamp(0.0, 150.0) / 150.0;
    }
    out[72] = s.mass as f32 / 3000.0;
    out[73] = s.boosting as u8 as f32;
    out[74] = (s.age as f32 / 14400.0).min(1.0);
    out[75] = s
        .crown_at
        .map(|t| (w.tick - t) as f32 / 2700.0)
        .unwrap_or(0.0);
    out
}
#[pymethods]
impl Batch {
    #[new]
    #[pyo3(signature=(worlds=1, agents=8, seed=1))]
    fn new(worlds: usize, agents: usize, seed: u64) -> PyResult<Self> {
        if !(1..=64).contains(&worlds) || !(1..=24).contains(&agents) {
            return Err(PyValueError::new_err(
                "worlds must be 1..64 and agents 1..24",
            ));
        }
        let mut result = Self {
            worlds: Vec::new(),
            agents: Vec::new(),
            slots: agents,
        };
        for n in 0..worlds {
            let mut w = World::new(seed.wrapping_add(n as u64));
            let ids = (0..agents)
                .filter_map(|i| w.spawn("synthetic", i))
                .collect::<Vec<_>>();
            if ids.len() != agents {
                return Err(PyValueError::new_err("failed to allocate all agent slots"));
            }
            result.worlds.push(w);
            result.agents.push(ids);
        }
        Ok(result)
    }
    fn observations(&self) -> Vec<Vec<f32>> {
        self.worlds
            .iter()
            .zip(&self.agents)
            .flat_map(|(w, ids)| ids.iter().map(move |&id| observation(w, id)))
            .collect()
    }
    /// 10 actions: five relative turns [-1.2,-0.6,0,.6,1.2], boost off/on.
    /// One decision advances three 30 Hz ticks. Dead slots respawn after terminal obs.
    fn step(&mut self, actions: Vec<usize>) -> PyResult<TransitionBatch> {
        if actions.len() != self.worlds.len() * self.slots || actions.iter().any(|&a| a >= 10) {
            return Err(PyValueError::new_err("expected one action 0..9 per slot"));
        }
        Ok(self.advance(&actions))
    }
    fn hashes(&self) -> Vec<u64> {
        self.worlds.iter().map(World::state_hash).collect()
    }
    fn mass_totals(&self) -> Vec<u32> {
        self.worlds.iter().map(World::total_mass).collect()
    }
}
impl Batch {
    fn advance(&mut self, actions: &[usize]) -> TransitionBatch {
        let mut obs = Vec::new();
        let mut rewards = Vec::new();
        let mut done = Vec::new();
        for (n, (w, ids)) in self.worlds.iter_mut().zip(&mut self.agents).enumerate() {
            let before: Vec<_> = ids
                .iter()
                .map(|id| w.snakes.get(id).map(|s| s.mass).unwrap_or(100))
                .collect();
            for (slot, &id) in ids.iter().enumerate() {
                if let Some(s) = w.snakes.get(&id) {
                    let a = actions[n * self.slots + slot];
                    w.action(
                        id,
                        Action {
                            heading: s.heading + [-1.2, -0.6, 0.0, 0.6, 1.2][a % 5],
                            boost: a >= 5,
                        },
                    );
                }
            }
            for _ in 0..3 {
                w.step();
            }
            for (slot, id) in ids.iter_mut().enumerate() {
                let dead = !w.snakes.contains_key(id);
                let mass = w.snakes.get(id).map(|s| s.mass).unwrap_or(0);
                rewards.push(if dead {
                    -1.0
                } else {
                    (mass as f32 / before[slot] as f32).ln() - 0.001
                });
                done.push(dead);
                obs.push(observation(w, *id));
            }
            // Capture every actor's transition from the same pre-reset world.
            // Respawns are visible only through the next observations() call.
            for (slot, id) in ids.iter_mut().enumerate() {
                if !w.snakes.contains_key(id)
                    && let Some(next) = w.spawn("synthetic", slot)
                {
                    *id = next;
                }
            }
        }
        (obs, rewards, done)
    }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Batch>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_observations_precede_all_respawns() {
        for seed in 1..=16 {
            let mut world = World::new(seed);
            let ids = (0..8)
                .map(|i| world.spawn("synthetic", i).unwrap())
                .collect();
            let mut batch = Batch {
                worlds: vec![world],
                agents: vec![ids],
                slots: 8,
            };
            batch.worlds[0].remove(batch.agents[0][0]);
            let mut expected_world = batch.worlds[0].clone();
            for _ in 0..3 {
                expected_world.step();
            }
            let expected: Vec<_> = batch.agents[0]
                .iter()
                .map(|&id| observation(&expected_world, id))
                .collect();
            let (actual, _, done) = batch.advance(&[2; 8]);
            assert!(done[0]);
            assert_eq!(actual, expected);
            assert!(batch.worlds[0].snakes.contains_key(&batch.agents[0][0]));
        }
    }
}
