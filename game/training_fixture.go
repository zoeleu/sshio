package game

import (
	"fmt"
	"math"
)

// PhysicsTrace provides controlled cross-backend fixtures. It is only exposed
// by the local trainenv command, not by the SSH game server.
type PhysicsState struct {
	Radius       float64     `json:"radius"`
	Tick         uint64      `json:"tick"`
	Snakes       []Snake     `json:"snakes"`
	Pellets      []Pellet    `json:"pellets"`
	Observations [][]float64 `json:"observations"`
}
type PhysicsTrace struct {
	Initial PhysicsState   `json:"initial"`
	Steps   []PhysicsState `json:"steps"`
}

func TrainingPhysicsTrace(name string, actions [][]int) (PhysicsTrace, error) {
	w := newSeededWorld(42, 80, 0, 0)
	add := func(p Vec2, h, score float64) *Snake {
		id := w.spawn("fixture", false)
		s := w.snakes[id]
		placeTrainingSnake(s, p, h, score)
		return s
	}
	a := add(Vec2{}, 0, 30)
	b := add(Vec2{30, 30}, 0, 30)
	switch name {
	case "straight":
	case "turn":
		a.Heading = 0.2
		a.TargetHeading = 0.2
	case "growth":
		w.pellets = []Pellet{{Vec2{0.8, 0}, 5}, {Vec2{2.4, 0}, 3}}
	case "boost":
		w.pellets = []Pellet{{Vec2{2, 0}, 1}}
	case "head_on":
		placeTrainingSnake(a, Vec2{-1.5, 0}, 0, 30)
		placeTrainingSnake(b, Vec2{1.5, 0}, math.Pi, 30)
	case "body":
		placeTrainingSnake(b, Vec2{2, 5}, math.Pi/2, 30)
	case "border":
		placeTrainingSnake(a, Vec2{79.9, 0}, 0, 30)
	case "minimum_boost":
		placeTrainingSnake(a, Vec2{}, 0, MinScore)
	case "own_body":
		a.Segments = append(a.Segments, Vec2{0.8, 0})
	case "contested_food":
		placeTrainingSnake(a, Vec2{-1, 0}, 0, 30)
		placeTrainingSnake(b, Vec2{1, 1.5}, math.Pi, 30)
		w.pellets = []Pellet{{Vec2{0, 0.75}, 3}}
	default:
		return PhysicsTrace{}, fmt.Errorf("unknown fixture %q", name)
	}
	stacks := map[int]*frameStack{}
	for _, id := range w.snakeIDs() {
		stacks[id] = &frameStack{}
	}
	state := func() PhysicsState {
		snapshot := w.Snapshot()
		v := PhysicsState{Radius: w.radius, Tick: w.tick, Snakes: snapshot.Snakes, Pellets: snapshot.Pellets}
		for _, id := range w.snakeIDs() {
			v.Observations = append(v.Observations, stacks[id].push(w.observeSingle(w.snakes[id])))
		}
		return v
	}
	// Snapshot ordering is by score, while observations/action slots are by ID.
	out := PhysicsTrace{Initial: state()}
	for _, batch := range actions {
		if len(batch) != 2 {
			return PhysicsTrace{}, fmt.Errorf("fixtures need two actions")
		}
		for i, action := range batch {
			if action < 0 || action >= ActionCount {
				return PhysicsTrace{}, fmt.Errorf("invalid fixture action")
			}
			w.applyAction(w.snakes[i+1], action, 4)
		}
		for tick := 0; tick < 4; tick++ {
			w.Tick()
		}
		out.Steps = append(out.Steps, state())
	}
	return out, nil
}
