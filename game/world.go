package game

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

const TPS = 20

type World struct {
	mu                   sync.RWMutex
	snakes               map[int]*Snake
	pellets              []Pellet
	subscribers          map[int]chan struct{}
	nextID               int
	tick                 uint64
	radius               float64
	target, pelletTarget int
	rng                  *rand.Rand
	policy               *Policy
	policyStacks         map[int]*frameStack
}

func NewWorld() *World { return newWorld(300, 15, 3500) }
func newWorld(radius float64, target, pellets int) *World {
	return newSeededWorld(time.Now().UnixNano(), radius, target, pellets)
}
func newSeededWorld(seed int64, radius float64, target, pellets int) *World {
	w := &World{snakes: make(map[int]*Snake), subscribers: make(map[int]chan struct{}), radius: radius, target: target, pelletTarget: pellets, rng: rand.New(rand.NewSource(seed))}
	w.balance()
	w.replenish()
	return w
}
func (w *World) Run(ctx context.Context) {
	t := time.NewTicker(time.Second / TPS)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			w.Tick()
		}
	}
}
func (w *World) spawn(name string, bot bool) int {
	w.nextID++
	id := w.nextID
	p := Vec2{}
	for attempt := 0; attempt < 100; attempt++ {
		a := w.rng.Float64() * 2 * math.Pi
		p = direction(a).Scale(math.Sqrt(w.rng.Float64()) * (w.radius - 30))
		safe := true
		for _, s := range w.snakes {
			if s.Alive {
				for _, b := range s.Segments {
					if b.Sub(p).Len() < 20 {
						safe = false
						break
					}
				}
			}
		}
		if safe {
			break
		}
	}
	h := w.rng.Float64() * 2 * math.Pi
	personality := PersonalityBalanced
	if bot {
		personality = (id % PersonalityCount) + 1
	}
	s := &Snake{ID: id, Personality: personality, Name: name, Heading: h, TargetHeading: h, Score: MinScore, Color: []int{45, 81, 213, 207, 118, 220, 209}[id%7], Bot: bot, Alive: true, speed: 0.8, prevHeading: h, headingInit: true}
	for i := 0; i < s.Length(); i++ {
		s.Segments = append(s.Segments, p.Sub(direction(h).Scale(float64(i)*Spacing)))
	}
	w.snakes[id] = s
	return id
}
func (w *World) Join(name string) (int, <-chan struct{}) {
	w.mu.Lock()
	defer w.mu.Unlock()
	id := w.spawn(name, false)
	ch := make(chan struct{}, 1)
	w.subscribers[id] = ch
	w.balance()
	return id, ch
}
func (w *World) Spectate() (int, <-chan struct{}) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.nextID++
	id := w.nextID
	ch := make(chan struct{}, 1)
	w.subscribers[id] = ch
	return id, ch
}
func (w *World) Leave(id int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if s := w.snakes[id]; s != nil && s.Alive {
		w.kill(s, "disconnected")
	}
	delete(w.snakes, id)
	if ch, ok := w.subscribers[id]; ok {
		close(ch)
		delete(w.subscribers, id)
	}
	w.balance()
}
func (w *World) SetInput(id int, heading float64, boost bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if s := w.snakes[id]; s != nil {
		if !math.IsNaN(heading) && !math.IsInf(heading, 0) {
			s.TargetHeading = heading
		}
		if boost {
			s.BoostUntil = w.tick + 5
		}
	}
}
func (w *World) SetMouseBoost(id int, boost bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if s := w.snakes[id]; s != nil {
		s.MouseBoost = boost
	}
}
func (w *World) Snapshot() Snapshot {
	w.mu.RLock()
	defer w.mu.RUnlock()
	v := Snapshot{Radius: w.radius, Tick: w.tick, Pellets: append([]Pellet(nil), w.pellets...)}
	for _, s := range w.snakes {
		c := *s
		c.Segments = append([]Vec2(nil), s.Segments...)
		v.Snakes = append(v.Snakes, c)
	}
	sort.Slice(v.Snakes, func(i, j int) bool {
		if v.Snakes[i].Score == v.Snakes[j].Score {
			return v.Snakes[i].ID < v.Snakes[j].ID
		}
		return v.Snakes[i].Score > v.Snakes[j].Score
	})
	return v
}
func (w *World) kill(s *Snake, killer string) {
	s.Alive = false
	s.Killer = killer
	for i, p := range s.Segments {
		if i%3 == 0 {
			w.pellets = append(w.pellets, Pellet{p, 2})
		}
	}
}
func (w *World) replenish() {
	if w.pelletTarget <= 0 {
		return
	}
	totalMass := 0.0
	for _, s := range w.snakes {
		if s.Alive && s.Score > MinScore {
			totalMass += s.Score - MinScore
		}
	}
	for _, p := range w.pellets {
		totalMass += p.Value
	}
	budget := float64(w.pelletTarget)
	maxByCount := w.pelletTarget - len(w.pellets)
	maxByMass := int(math.Floor(budget - totalMass))
	floorNeeded := w.pelletTarget/4 - len(w.pellets)
	needed := max(0, max(floorNeeded, min(maxByCount, maxByMass)))
	for i := 0; i < needed; i++ {
		p := direction(w.rng.Float64() * 2 * math.Pi).Scale(math.Sqrt(w.rng.Float64()) * (w.radius - 2))
		w.pellets = append(w.pellets, Pellet{p, 1})
	}
}
func (w *World) balance() {
	count := 0
	bots := []int{}
	for id, s := range w.snakes {
		if s.Alive {
			count++
			if s.Bot {
				bots = append(bots, id)
			}
		} else if s.Bot {
			delete(w.snakes, id)
		}
	}
	sort.Ints(bots)
	for count > w.target && len(bots) > 0 {
		id := bots[len(bots)-1]
		bots = bots[:len(bots)-1]
		delete(w.snakes, id)
		count--
	}
	for count < w.target {
		w.spawn(fmt.Sprintf("%s", botNames[w.nextID%len(botNames)]), true)
		count++
	}
}
func (w *World) Tick() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.tick++
	ordered := w.snakeIDs()
	bodies := grid{}
	for _, id := range ordered {
		s := w.snakes[id]
		if s.Alive {
			for _, p := range s.Segments {
				bodies.add(bodyPoint{p, s.ID})
			}
		}
	}
	for _, id := range ordered {
		s := w.snakes[id]
		if s.Alive && s.Bot {
			if w.policy == nil {
				w.steerBot(s, bodies)
			} else if (w.tick+uint64(s.ID))%4 == 0 {
				w.applyAction(s, w.policy.PersonalityAction(w.policyObservation(s), s.Personality), 4)
			}
		}
	}
	// Substeps keep fast heads from tunneling through bodies or each other.
	for step := 0; step < 4; step++ {
		old := map[int]Vec2{}
		for _, id := range ordered {
			s := w.snakes[id]
			if !s.Alive {
				continue
			}
			old[id] = s.Segments[0]
			s.Heading = turn(s.Heading, s.TargetHeading, 0.18/4)
			speed := 0.8 / 4
			boost := (s.MouseBoost || s.BoostUntil > w.tick) && s.Score > MinScore
			if boost {
				speed *= 2
				if step == 0 {
					spent := math.Min(0.2, s.Score-MinScore)
					s.BoostSpent += spent
					s.Score = math.Max(MinScore, s.Score-0.2)
					if w.tick%4 == 0 {
						w.pellets = append(w.pellets, Pellet{s.Segments[len(s.Segments)-1], 0.4})
					}
				}
			}
			s.speed, s.boosting = speed*4, boost
			s.Segments[0] = s.Segments[0].Add(direction(s.Heading).Scale(speed))
			for i := 1; i < len(s.Segments); i++ {
				delta := s.Segments[i-1].Sub(s.Segments[i])
				if d := delta.Len(); d > Spacing {
					s.Segments[i] = s.Segments[i-1].Sub(delta.Scale(Spacing / d))
				}
			}
		}
		g := grid{}
		for _, id := range ordered {
			s := w.snakes[id]
			if s.Alive {
				for _, p := range s.Segments {
					g.add(bodyPoint{p, s.ID})
				}
			}
		}
		type death struct {
			name string
			id   int
		}
		dead := map[int]death{}
		for _, id := range ordered {
			if _, ok := old[id]; !ok {
				continue
			}
			s := w.snakes[id]
			h := s.Segments[0]
			if h.Len() > w.radius {
				dead[id] = death{name: "the border"}
			}
			searchR := 0.425 * (s.WidthScale() + 4.0)
			g.near(h, searchR, func(p bodyPoint) {
				if p.id != id {
					hitR := 0.85
					if other := w.snakes[p.id]; other != nil {
						hitR = 0.425 * (s.WidthScale() + other.WidthScale())
					}
					if h.Sub(p.pos).Len() < hitR {
						dead[id] = death{w.snakes[p.id].Name, p.id}
					}
				}
			})
		}
		for _, id := range ordered {
			k, ok := dead[id]
			if !ok {
				continue
			}
			other, mutual := dead[k.id]
			if !mutual || other.id != id {
				continue
			}
			a, b := w.snakes[id], w.snakes[k.id]
			if a == nil || b == nil {
				continue
			}
			if a.boosting != b.boosting {
				if a.boosting {
					delete(dead, id)
				} else {
					delete(dead, k.id)
				}
			} else if a.Score != b.Score {
				if a.Score > b.Score {
					delete(dead, id)
				} else {
					delete(dead, k.id)
				}
			}
		}
		for _, id := range ordered {
			if k, ok := dead[id]; ok {
				if killer := w.snakes[k.id]; killer != nil {
					killer.Kills++
					if _, mutual := dead[k.id]; mutual {
						killer.MutualKills++
					}
				}
				w.kill(w.snakes[id], k.name)
			}
		}
	}
	for _, id := range ordered {
		s := w.snakes[id]
		if s == nil || !s.headingInit {
			continue
		}
		s.turnRate = clamp(math.Atan2(math.Sin(s.Heading-s.prevHeading), math.Cos(s.Heading-s.prevHeading)) / 0.18)
		s.prevTurn = clamp(s.turnRate * 0.18 / 0.9)
		if s.boosting {
			s.prevBoost = 1
		} else {
			s.prevBoost = 0
		}
		s.prevHeading = s.Heading
	}
	// Pellet hash avoids scanning every pellet for every head.
	pg := grid{}
	for i, p := range w.pellets {
		pg.add(bodyPoint{p.Pos, i})
	}
	eaten := map[int]bool{}
	ids := make([]int, 0, len(w.snakes))
	for id := range w.snakes {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	for _, id := range ids {
		s := w.snakes[id]
		if !s.Alive {
			continue
		}
		pickupR := 1.3 + 0.45*(s.WidthScale()-1.0)
		rawGain := 0.0
		pg.near(s.Segments[0], pickupR, func(p bodyPoint) {
			if !eaten[p.id] && s.Segments[0].Sub(p.pos).Len() < pickupR {
				rawGain += w.pellets[p.id].Value
				eaten[p.id] = true
			}
		})
		if rawGain > 0 {
			eff := 1.0 / (1.0 + math.Max(0, s.Score-60.0)/120.0)
			s.Score += rawGain * eff
			s.FoodEaten += rawGain
		}
		if s.Score > 120.0 {
			s.Score -= 0.002 * (s.Score - 120.0)
		}
		for len(s.Segments) < s.Length() {
			s.Segments = append(s.Segments, s.Segments[len(s.Segments)-1])
		}
		s.Segments = s.Segments[:s.Length()]
	}
	kept := w.pellets[:0]
	for i, p := range w.pellets {
		if !eaten[i] {
			kept = append(kept, p)
		}
	}
	w.pellets = kept
	w.balance()
	w.replenish()
	for _, ch := range w.subscribers {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// Stable ordering makes seeded training episodes reproducible. Caller holds the lock.
func (w *World) snakeIDs() []int {
	ids := make([]int, 0, len(w.snakes))
	for id := range w.snakes {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}
