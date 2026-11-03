package game

import "math"

var botNames = []string{"Noodle", "Miso", "Ziggy", "Orbit", "Pickle", "Comet", "Squig", "Mochi", "Fern", "Pebble", "Wisp", "Tango", "Kelp", "Pip", "Ramen"}

// Goals survive steering updates, so a bot finishes a meal or a cut-off rather
// than switching between equally attractive targets on every tick.
type botBrain struct {
	goal  Vec2
	until uint64
	prey  int
}

func angleTo(from, to Vec2) float64      { d := to.Sub(from); return math.Atan2(d.Y, d.X) }
func angleDistance(a, b float64) float64 { return math.Abs(math.Atan2(math.Sin(a-b), math.Cos(a-b))) }

type foodTarget struct {
	pos   Vec2
	value float64
	dist  float64
}

// foodTargets returns up to n pellets ranked by the heuristic utility used
// for the primary food goal, best first.
func (w *World) foodTargets(s *Snake, n int) []foodTarget {
	h := s.Segments[0]
	type patch struct{ x, y int }
	patches := map[patch]float64{}
	key := func(p Vec2) patch { return patch{int(math.Floor((p.X + 0.001) / 12)), int(math.Floor((p.Y + 0.001) / 12))} }
	for _, p := range w.pellets {
		patches[key(p.Pos)] += p.Value
	}
	targets := []foodTarget{}
	utilities := []float64{}
	for _, p := range w.pellets {
		d := p.Pos.Sub(h).Len()
		a := angleDistance(s.Heading, angleTo(h, p.Pos))
		if p.Pos.Len() > w.radius-5 || (d < 9 && a > math.Pi/2) {
			continue
		}
		value := p.Value + math.Min(15, patches[key(p.Pos)])*0.3
		utility := value / (8 + d + 12*a)
		// Insert into a best-first list capped at n.
		i := 0
		for i < len(targets) && utilities[i] >= utility {
			i++
		}
		if i >= n {
			continue
		}
		targets = append(targets, foodTarget{})
		utilities = append(utilities, 0)
		copy(targets[i+1:], targets[i:])
		copy(utilities[i+1:], utilities[i:])
		targets[i] = foodTarget{pos: p.Pos, value: value, dist: d}
		utilities[i] = utility
		if len(targets) > n {
			targets, utilities = targets[:n], utilities[:n]
		}
	}
	for i := range targets {
		if targets[i].pos.Len() > w.radius-10 {
			targets[i].pos = h.Scale(0.5)
		}
	}
	return targets
}

func (w *World) foodGoal(s *Snake) Vec2 {
	h := s.Segments[0]
	// Reward patches of food as well as individual high-value death pellets.
	type patch struct{ x, y int }
	patches := map[patch]float64{}
	key := func(p Vec2) patch { return patch{int(math.Floor((p.X + 0.001) / 12)), int(math.Floor((p.Y + 0.001) / 12))} }
	for _, p := range w.pellets {
		patches[key(p.Pos)] += p.Value
	}
	best := math.Inf(-1)
	goal := h.Add(direction(s.Heading).Scale(30))
	for _, p := range w.pellets {
		d := p.Pos.Sub(h).Len()
		a := angleDistance(s.Heading, angleTo(h, p.Pos))
		// A nearby pellet behind the turning circle is a trap: come back later.
		if p.Pos.Len() > w.radius-5 || (d < 9 && a > math.Pi/2) {
			continue
		}
		utility := (p.Value + math.Min(15, patches[key(p.Pos)])*0.3) / (8 + d + 12*a)
		if utility > best {
			best = utility
			goal = p.Pos
		}
	}
	if goal.Len() > w.radius-10 {
		goal = h.Scale(0.5)
	}
	return goal
}

// Attack the space ahead of a rival, laying our body across its route. Chasing
// the head itself only causes mutual deaths. Only take races we can win.
func (w *World) attackGoal(s *Snake) (Vec2, int) {
	if s.Score < MinScore+8 {
		return Vec2{}, 0
	}
	h := s.Segments[0]
	best := math.Inf(1)
	goal := Vec2{}
	prey := 0
	for _, id := range w.snakeIDs() {
		other := w.snakes[id]
		if !other.Alive || other.ID == s.ID {
			continue
		}
		head := other.Segments[0]
		d := head.Sub(h).Len()
		if d < 7 || d > 45 {
			continue
		}
		forward := direction(other.Heading)
		lead := math.Max(8, math.Min(22, d*0.8))
		crossing := head.Add(forward.Scale(lead))
		// Stop pursuing a rival moving away from us with no intercept advantage.
		arrival := crossing.Sub(h).Len()/1.6 + angleDistance(s.Heading, angleTo(h, crossing))/0.18
		rivalSpeed := 0.8
		if other.MouseBoost || other.BoostUntil > w.tick {
			rivalSpeed = 1.6
		}
		if arrival+3 > lead/rivalSpeed {
			continue
		}
		normal := Vec2{-forward.Y, forward.X}
		side := h.Sub(crossing).X*normal.X + h.Sub(crossing).Y*normal.Y
		if side > 0 {
			normal = normal.Scale(-1)
		}
		target := crossing.Add(normal.Scale(4))
		if target.Len() > w.radius-12 {
			continue
		}
		if arrival < best || (arrival == best && other.ID < prey) {
			best = arrival
			goal = target
			prey = other.ID
		}
	}
	return goal, prey
}

func (w *World) steerBot(s *Snake, g grid) {
	h := s.Segments[0]
	brain := &s.brain
	valid := false
	if brain.prey != 0 {
		// Replan moving interceptions regularly, but keep the crossing point steady
		// between plans so the bot actually crosses the opponent's lane.
		p := w.snakes[brain.prey]
		valid = p != nil && p.Alive
	} else {
		for _, p := range w.pellets {
			if p.Pos == brain.goal {
				valid = true
				break
			}
		}
	}
	if !valid || w.tick >= brain.until || h.Sub(brain.goal).Len() < 1.3 {
		brain.goal = w.foodGoal(s)
		brain.prey = 0
		brain.until = w.tick + 60
		if goal, prey := w.attackGoal(s); prey != 0 {
			brain.goal = goal
			brain.prey = prey
			brain.until = w.tick + 12
		}
	} else if brain.prey == 0 && w.tick%10 == uint64(s.ID%10) {
		// A nearby opening can interrupt foraging without abandoning every meal.
		if goal, prey := w.attackGoal(s); prey != 0 {
			brain.goal = goal
			brain.prey = prey
			brain.until = w.tick + 12
		}
	}
	desired := angleTo(h, brain.goal)
	speed := 0.8
	boost := brain.prey != 0 && s.Score > MinScore+5 && angleDistance(s.Heading, desired) < 0.45
	if boost {
		speed = 1.6
	}
	// Evaluate curved paths using the actual turn limit. Straight rays miss
	// obstacles while turning, and coarse left/straight/right choices orbit food.
	opponents := w.snakeIDs()
	bestHeading := s.Heading
	bestRisk := math.Inf(1)
	for attempt := 0; attempt < 2; attempt++ {
		bestCost := math.Inf(1)
		bestRisk = math.Inf(1)
		for _, a := range []float64{desired, s.Heading, s.Heading - 0.3, s.Heading + 0.3, s.Heading - 0.7, s.Heading + 0.7, s.Heading - 1.3, s.Heading + 1.3} {
			pos, heading := h, s.Heading
			risk := 0.0
			for tick := 1; tick <= 18; tick++ {
				heading = turn(heading, a, 0.18)
				// Half-tick probes also cover the faster boosted path.
				for sub := 0; sub < 2; sub++ {
					pos = pos.Add(direction(heading).Scale(speed / 2))
					clearance := w.radius - pos.Len()
					if clearance < 2 {
						risk += 1000
					} else if clearance < 7 {
						risk += (7 - clearance) * 3
					}
					nearest := 4.0
					g.near(pos, 4, func(b bodyPoint) {
						if b.id != s.ID {
							nearest = math.Min(nearest, b.pos.Sub(pos).Len())
						}
					})
					if nearest < 1.3 {
						risk += 1000
					} else if nearest < 3 {
						risk += (3 - nearest) * 8
					}
				}
				// Bodies are covered by the hash; predict approaching heads as well.
				for _, id := range opponents {
					other := w.snakes[id]
					if other.ID != s.ID && other.Alive {
						v := 0.8
						if other.MouseBoost || other.BoostUntil > w.tick {
							v = 1.6
						}
						predicted := other.Segments[0].Add(direction(other.Heading).Scale(v * float64(tick)))
						if d := predicted.Sub(pos).Len(); d < 2.5 {
							risk += (2.5 - d) * 40
						}
					}
				}
			}
			cost := risk + pos.Sub(brain.goal).Len() + angleDistance(a, desired)*2
			if cost < bestCost {
				bestCost = cost
				bestHeading = a
				bestRisk = risk
			}
		}
		if !boost || (bestRisk < 1 && angleDistance(bestHeading, desired) < 0.2) {
			break
		}
		// If the sprint is unsafe, evaluate the slower turning path instead.
		boost = false
		speed = 0.8
	}
	s.TargetHeading = bestHeading
	// Do not keep a previous burst alive when evasive steering is necessary.
	s.BoostUntil = w.tick
	if boost && bestRisk < 1 && angleDistance(bestHeading, desired) < 0.2 {
		s.BoostUntil = w.tick + 2
	}
}
