package game

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type trainingArena struct {
	world        *World
	ids          []int
	age, episode int
	seed         int64
	index        int
	scenario     string
	motion       []motionTracker
	stacks       []*frameStack
}
type TrainingBatch struct {
	arenas          []*trainingArena
	agents, horizon int
	curriculum      bool
	stage           int
}
type BatchTransition struct {
	Observations     [][]float64 `json:"observations"`
	Rewards          []float64   `json:"rewards"`
	Dones            []bool      `json:"dones"`
	Teachers         []int       `json:"teachers,omitempty"`
	Food             float64     `json:"food"`
	Kills            int         `json:"kills"`
	Deaths           int         `json:"deaths"`
	MutualKills      int         `json:"mutual_kills"`
	Circles          int         `json:"circles"`
	AgentFood        []float64   `json:"agent_food"`
	AgentKills       []int       `json:"agent_kills"`
	AgentMutualKills []int       `json:"agent_mutual_kills"`
	AgentDeaths      []bool      `json:"agent_deaths"`
	AgentCircles     []int       `json:"agent_circles"`
	Scenarios        []string    `json:"scenarios"`
	Radii            []float64   `json:"radii"`
}

func NewTrainingBatch(seed int64, arenas, agents, horizon int) (*TrainingBatch, error) {
	if arenas < 1 || arenas > 64 || agents < 2 || agents > 64 || arenas*agents > 2048 || horizon < 4 || horizon > 20000 {
		return nil, fmt.Errorf("need 1..64 arenas, 2..64 agents, <=2048 total worms, horizon 4..20000")
	}
	b := &TrainingBatch{agents: agents, horizon: horizon}
	for i := 0; i < arenas; i++ {
		a := &trainingArena{seed: seed + int64(i)*1000003, index: i}
		b.resetArena(a)
		b.arenas = append(b.arenas, a)
	}
	return b, nil
}

// SetCurriculum changes future episodes; reset=true applies the initial phase now.
func (b *TrainingBatch) SetCurriculum(stage int, reset bool) error {
	if stage < 0 || stage > 2 {
		return fmt.Errorf("curriculum stage must be 0..2")
	}
	b.curriculum = true
	b.stage = stage
	if reset {
		for _, a := range b.arenas {
			b.resetArena(a)
		}
	}
	return nil
}
func (b *TrainingBatch) resetArena(a *trainingArena) {
	scale := math.Max(1, float64(b.agents)/4)
	seed := a.seed + int64(a.episode)*7919
	radius, pellets := 90*math.Sqrt(scale), int(320*scale)
	a.scenario = "arena"
	if b.curriculum {
		scenarios := [][]string{{"forage", "border", "arena"}, {"forage", "border", "cutoff", "escape", "chase", "arena"}, {"forage", "border", "cutoff", "escape", "chase", "arena", "sparse", "dense", "production"}}
		options := scenarios[b.stage]
		a.scenario = options[(a.index+a.episode)%len(options)]
		rng := rand.New(rand.NewSource(seed ^ 0x51491))
		radius *= 0.8 + 0.4*rng.Float64()
		pellets = int(float64(pellets) * (0.6 + 0.8*rng.Float64()))
		if a.scenario == "sparse" {
			radius *= 1.4
		}
		if a.scenario == "dense" {
			radius *= 0.75
		}
		if a.scenario == "production" {
			radius = 300 * math.Sqrt(float64(b.agents)/15)
			pellets = 3500 * b.agents / 15
		}
	}
	a.world = newSeededWorld(seed, radius, 0, pellets)
	a.ids = nil
	a.age = 0
	a.episode++
	a.motion = make([]motionTracker, b.agents)
	a.stacks = make([]*frameStack, b.agents)
	for i := range a.stacks {
		a.stacks[i] = &frameStack{}
	}
	for i := 0; i < b.agents; i++ {
		a.ids = append(a.ids, a.world.spawn(fmt.Sprintf("worm-%d", i), false))
	}
	if b.curriculum {
		b.arrangeScenario(a)
	}
	for i, id := range a.ids {
		s := a.world.snakes[id]
		a.motion[i].sample(s.Segments[0], s.Heading)
	}
}
func (b *TrainingBatch) arrangeScenario(a *trainingArena) {
	w := a.world
	switch a.scenario {
	case "border":
		rotation := w.rng.Float64() * 2 * math.Pi
		for i, id := range a.ids {
			h := rotation + float64(i)*2*math.Pi/float64(b.agents)
			placeTrainingSnake(w.snakes[id], direction(h).Scale(w.radius-10), h, 20)
		}
	case "cutoff", "escape", "chase":
		pairs := (b.agents + 1) / 2
		cols := int(math.Ceil(math.Sqrt(float64(pairs))))
		rows := (pairs + cols - 1) / cols
		spacing := w.radius * 1.1 / float64(cols)
		for i := 0; i+1 < len(a.ids); i += 2 {
			pair := i / 2
			center := Vec2{(float64(pair%cols) - float64(cols-1)/2) * spacing, (float64(pair/cols) - float64(rows-1)/2) * spacing}
			h := w.rng.Float64() * 2 * math.Pi
			rotate := func(p Vec2) Vec2 {
				return center.Add(Vec2{p.X*math.Cos(h) - p.Y*math.Sin(h), p.X*math.Sin(h) + p.Y*math.Cos(h)})
			}
			p0, p1, h0, h1, score := Vec2{0, -10}, Vec2{-10, 0}, math.Pi/2, 0.0, 30.0
			if a.scenario == "escape" {
				p0, p1, score = Vec2{0, -7}, Vec2{-7, 0}, 45
			}
			if a.scenario == "chase" {
				p0, p1, h1, score = Vec2{0, -12}, Vec2{0, 10}, math.Pi/2, 45
			}
			score0, score1 := score, score
			if a.scenario == "escape" {
				score0, score1 = 20, 70
			}
			if a.scenario == "chase" {
				score1 = 20
			}
			placeTrainingSnake(w.snakes[a.ids[i]], rotate(p0), h+h0, score0)
			placeTrainingSnake(w.snakes[a.ids[i+1]], rotate(p1), h+h1, score1)
		}
	default:
		for _, id := range a.ids {
			s := w.snakes[id]
			score := MinScore
			if a.scenario != "forage" {
				score += w.rng.Float64() * 45
			}
			placeTrainingSnake(s, s.Segments[0], s.Heading, score)
			if a.scenario == "forage" {
				for i := 0; i < 12; i++ {
					w.pellets = append(w.pellets, Pellet{s.Segments[0].Add(direction(s.Heading).Scale(8 + float64(i))), 1})
				}
			}
		}
	}
}
func (b *TrainingBatch) State(teach bool) BatchTransition { return b.transition(nil, teach) }
func (b *TrainingBatch) Step(actions []int, teach bool) (BatchTransition, error) {
	if len(actions) != len(b.arenas)*b.agents {
		return BatchTransition{}, fmt.Errorf("wrong number of actions")
	}
	for _, a := range actions {
		if a < -1 || a >= ActionCount {
			return BatchTransition{}, fmt.Errorf("invalid action %d", a)
		}
	}
	return b.transition(actions, teach), nil
}
func (b *TrainingBatch) transition(actions []int, teach bool) BatchTransition {
	n := len(b.arenas) * b.agents
	out := BatchTransition{Observations: make([][]float64, n), Rewards: make([]float64, n), Dones: make([]bool, n), AgentFood: make([]float64, n), AgentKills: make([]int, n), AgentMutualKills: make([]int, n), AgentDeaths: make([]bool, n), AgentCircles: make([]int, n), Scenarios: make([]string, len(b.arenas)), Radii: make([]float64, len(b.arenas))}
	if teach {
		out.Teachers = make([]int, n)
	}
	type counts struct {
		food                           float64
		kills, deaths, mutual, circles int
	}
	stats := make([]counts, len(b.arenas))
	var wg sync.WaitGroup
	for index, a := range b.arenas {
		wg.Add(1)
		go func(index int, a *trainingArena) {
			defer wg.Done()
			w := a.world
			base := index * b.agents
			if actions != nil {
				scores := make([]float64, b.agents)
				kills := make([]int, b.agents)
				food := make([]float64, b.agents)
				mutual := make([]int, b.agents)
				for i, id := range a.ids {
					s := w.snakes[id]
					scores[i] = s.Score
					kills[i] = s.Kills
					food[i] = s.FoodEaten
					mutual[i] = s.MutualKills
					if actions[base+i] >= 0 {
						w.applyAction(s, actions[base+i], 4)
					}
				}
				for tick := 0; tick < 4 && a.age < b.horizon; tick++ {
					hasTeacher := false
					for i := range a.ids {
						if actions[base+i] == -1 {
							hasTeacher = true
							break
						}
					}
					if hasTeacher {
						g := grid{}
						for _, id := range a.ids {
							s := w.snakes[id]
							if s.Alive {
								for _, p := range s.Segments {
									g.add(bodyPoint{p, id})
								}
							}
						}
						for i, id := range a.ids {
							s := w.snakes[id]
							if s.Alive && actions[base+i] == -1 {
								w.steerBot(s, g)
							}
						}
					}
					w.Tick()
					a.age++
				}
				for i, id := range a.ids {
					s := w.snakes[id]
					gained := s.FoodEaten - food[i]
					out.AgentFood[base+i] = gained
					out.AgentMutualKills[base+i] = s.MutualKills - mutual[i]
					stats[index].mutual += out.AgentMutualKills[base+i]
					if a.motion[i].sample(s.Segments[0], s.Heading) {
						out.AgentCircles[base+i] = 1
						stats[index].circles++
					}
					k := s.Kills - kills[i]
					out.AgentKills[base+i] = k
					stats[index].food += gained
					stats[index].kills += k
					out.Rewards[base+i] = 0.3*(s.Score-scores[i]) + 6*float64(k) - 0.01
					if !s.Alive {
						out.Dones[base+i] = true
						out.Rewards[base+i] -= 10
						stats[index].deaths++
						out.AgentDeaths[base+i] = true
						a.motion[i] = motionTracker{}
						a.stacks[i].reset()
						delete(w.snakes, id)
						a.ids[i] = w.spawn(s.Name, false)
					}
				}
				// Treat the arena time limit as terminal; no value leaks into a reset.
				if a.age >= b.horizon {
					for i := 0; i < b.agents; i++ {
						out.Dones[base+i] = true
					}
					b.resetArena(a)
					w = a.world
				}
			}
			out.Scenarios[index] = a.scenario
			out.Radii[index] = w.radius
			for i, id := range a.ids {
				s := w.snakes[id]
				out.Observations[base+i] = a.stacks[i].push(w.observeSingle(s))
				if teach {
					env := TrainingEnv{world: w, id: id}
					out.Teachers[base+i] = env.teacher()
				}
			}
		}(index, a)
	}
	wg.Wait()
	for _, c := range stats {
		out.Food += c.food
		out.Kills += c.kills
		out.Deaths += c.deaths
		out.MutualKills += c.mutual
		out.Circles += c.circles
	}
	return out
}
