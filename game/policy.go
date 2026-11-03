package game

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Observation v2 layout (all egocentric, bounded to [-1, 1]):
//
//	Single frame (SingleObservationSize = 172):
//	  [0:16]   self/global: linear+log size, can-boost, border, centre
//	           direction, speed, boost, turn rate, previous turn/boost,
//	           length, nearest-rival distance, growth fraction, boost
//	           headroom, radial position.
//	  [16:32]  4 food targets x (dx, dy, distance, value), best first.
//	  [32:76]  4 nearest rivals x (dx, dy, vel-forward, vel-side,
//	           heading cos/sin, size ratio, distance, boosting,
//	           turn rate, presence), nearest first.
//	  [76:172] 24 sectors x (body clearance, head clearance,
//	           border range, food value).
//
// The policy input stacks FrameCount single frames, oldest first, so
// ObservationSize = FrameCount * SingleObservationSize = 688.
const SingleObservationSize = 172
const FrameCount = 4
const ObservationSize = FrameCount * SingleObservationSize
const PersonalityObservationSize = ObservationSize + PersonalityCount
const LegacyObservationSize = 49
const ActionCount = 14
const PolicyVersion = 2
const PersonalityPolicyVersion = 3
const LegacyPolicyVersion = 1

const (
	foodTargetsCount = 4
	rivalCount       = 4
	sectorCount      = 24
	foodBlock        = 16
	rivalBlock       = 32
	sectorBlock      = 76
	rivalStride      = 11
	sectorStride     = 4
)

var actionTurns = [...]float64{-0.9, -0.4, -0.15, 0, 0.15, 0.4, 0.9}

// Policy is an immutable tanh MLP. JSON weights are row-major [out][in].
// Version 1 (49 inputs, single frame) remains loadable as a baseline;
// version 2 consumes the stacked 688-input observation;
// version 3 consumes the stacked 688-input observation + 4-dim personality one-hot (692).
//
// The network is input -> tanh(W1) -> [tanh(WH)] -> W2 (logits). WH/BH are an
// optional second hidden layer; exports without them are single-hidden-layer
// policies and load unchanged. When present, W2 consumes the WH layer's width.
type Policy struct {
	Version int         `json:"version"`
	W1      [][]float64 `json:"w1"`
	B1      []float64   `json:"b1"`
	WH      [][]float64 `json:"wh,omitempty"`
	BH      []float64   `json:"bh,omitempty"`
	W2      [][]float64 `json:"w2"`
	B2      []float64   `json:"b2"`
}

// maxHiddenWidth bounds every hidden layer so a hostile or corrupt file cannot
// force huge allocations in the 20 TPS game loop.
const maxHiddenWidth = 256

// lastHiddenWidth is the width of the layer feeding the output layer.
func (p *Policy) lastHiddenWidth() int {
	if len(p.WH) > 0 {
		return len(p.WH)
	}
	return len(p.W1)
}

// InputSize is the observation length this policy consumes.
func (p *Policy) InputSize() int {
	if p.Version == LegacyPolicyVersion {
		return LegacyObservationSize
	}
	if p.Version == PersonalityPolicyVersion {
		return PersonalityObservationSize
	}
	return ObservationSize
}

func (p *Policy) Validate() error {
	if p.Version != PolicyVersion && p.Version != LegacyPolicyVersion && p.Version != PersonalityPolicyVersion {
		return fmt.Errorf("unsupported policy version %d", p.Version)
	}
	if len(p.W1) < 1 || len(p.W1) > maxHiddenWidth || len(p.B1) != len(p.W1) || len(p.W2) != ActionCount || len(p.B2) != ActionCount {
		return fmt.Errorf("invalid policy dimensions")
	}
	if len(p.WH) != len(p.BH) || len(p.WH) > maxHiddenWidth {
		return fmt.Errorf("invalid second hidden layer dimensions")
	}
	layers := [][][]float64{p.W1}
	widths := []int{p.InputSize()}
	if len(p.WH) > 0 {
		layers, widths = append(layers, p.WH), append(widths, len(p.W1))
	}
	layers, widths = append(layers, p.W2), append(widths, p.lastHiddenWidth())
	for layer, weights := range layers {
		width := widths[layer]
		for _, row := range weights {
			if len(row) != width {
				return fmt.Errorf("invalid layer %d width", layer)
			}
			for _, v := range row {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return fmt.Errorf("non-finite weight")
				}
			}
		}
	}
	for _, biases := range [][]float64{p.B1, p.BH, p.B2} {
		for _, v := range biases {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("non-finite bias")
			}
		}
	}
	return nil
}
func LoadPolicy(path string) (*Policy, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var p Policy
	d := json.NewDecoder(io.LimitReader(f, 8<<20))
	d.DisallowUnknownFields()
	if err = d.Decode(&p); err != nil {
		return nil, err
	}
	if err = p.Validate(); err != nil {
		return nil, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("unexpected data after policy")
	}
	return &p, nil
}
func (p *Policy) Action(obs []float64) int {
	return p.PersonalityAction(obs, PersonalityBalanced)
}

// PersonalityAction evaluates the policy for a specific bot personality.
// For version 3 policies, the 4-element personality one-hot vector is appended
// (or read from obs[688:692]). For version 2 policies, observation/logit
// conditioning modulates the forward pass according to the personality.
func (p *Policy) PersonalityAction(obs []float64, personality int) int {
	if p.Version == PersonalityPolicyVersion && len(obs) == ObservationSize {
		extended := make([]float64, PersonalityObservationSize)
		copy(extended, obs)
		if personality >= PersonalityAggressive && personality <= PersonalityAfraid {
			extended[ObservationSize+personality-1] = 1
		}
		obs = extended
	}
	if len(obs) != p.InputSize() {
		return 3
	}
	if p.Version == PolicyVersion && personality != PersonalityBalanced {
		obs = modulateObservationV2(obs, personality)
	}
	hidden := tanhLayer(p.W1, p.B1, obs)
	if len(p.WH) > 0 {
		hidden = tanhLayer(p.WH, p.BH, hidden)
	}
	best, score := 0, math.Inf(-1)
	for i, row := range p.W2 {
		v := p.B2[i]
		for j, w := range row {
			v += w * hidden[j]
		}
		if p.Version == PolicyVersion && personality != PersonalityBalanced {
			v += personalityLogitBias(obs, personality, i)
		}
		if v > score {
			best = i
			score = v
		}
	}
	return best
}

// tanhLayer computes tanh(W*in + b) for a row-major [out][in] weight matrix.
func tanhLayer(weights [][]float64, biases, in []float64) []float64 {
	out := make([]float64, len(biases))
	for i, row := range weights {
		v := biases[i]
		for j, w := range row {
			v += w * in[j]
		}
		out[i] = math.Tanh(v)
	}
	return out
}

func modulateObservationV2(obs []float64, personality int) []float64 {
	out := append([]float64(nil), obs...)
	for frame := 0; frame < FrameCount; frame++ {
		base := frame * SingleObservationSize
		radialPos := out[base+15]
		borderClear := out[base+3]
		// In the safe interior of the arena (well away from the border), suppress the
		// unconditional unit-vector pull toward (0,0) that causes bots to clump in the middle.
		// Very close to (0,0), softly invert the center vector so bots roam outward.
		if borderClear >= 0.95 && radialPos < 0.78 {
			if radialPos < 0.22 {
				out[base+4] = clamp(-out[base+4] * 0.45)
				out[base+5] = clamp(-out[base+5] * 0.45)
			} else {
				out[base+4] *= 0.1
				out[base+5] *= 0.1
			}
			out[base+15] = 0.25
		}
		switch personality {
		case PersonalityGreedy:
			for f := 0; f < foodTargetsCount; f++ {
				fb := base + foodBlock + f*4
				out[fb+3] = clamp(out[fb+3] * 1.6)
			}
			for r := 0; r < sectorCount; r++ {
				sb := base + sectorBlock + r*sectorStride
				out[sb+3] = clamp(out[sb+3] * 1.5)
			}
		case PersonalityAfraid:
			out[base+12] = clamp(out[base+12] * 0.55)
			for r := 0; r < rivalCount; r++ {
				rb := base + rivalBlock + r*rivalStride
				if out[rb+10] > 0 {
					out[rb+6] = clamp(out[rb+6] * 1.5)
					out[rb+7] = clamp(out[rb+7] * 0.55)
				}
			}
			for s := 0; s < sectorCount; s++ {
				sb := base + sectorBlock + s*sectorStride
				out[sb] = clamp(out[sb] * 0.65)
				out[sb+1] = clamp(out[sb+1] * 0.5)
			}
		case PersonalityConservative:
			out[base+2] = 0 // never boost
			for s := 0; s < sectorCount; s++ {
				sb := base + sectorBlock + s*sectorStride
				out[sb] = clamp(out[sb] * 0.8)
				out[sb+1] = clamp(out[sb+1] * 0.75)
			}
		case PersonalityAggressive:
			for r := 0; r < rivalCount; r++ {
				rb := base + rivalBlock + r*rivalStride
				if out[rb+10] > 0 {
					out[rb+6] = clamp(out[rb+6] * 0.7) // rivals appear smaller/prey-like
				}
			}
		}
	}
	return out
}

func personalityLogitBias(obs []float64, personality, action int) float64 {
	newest := (FrameCount - 1) * SingleObservationSize
	canBoost := obs[newest+2] > 0
	nearRivalDist := obs[newest+12]
	switch personality {
	case PersonalityConservative:
		if action >= 7 {
			return -10.0
		}
	case PersonalityAfraid:
		if action >= 7 && nearRivalDist < 0.35 && canBoost {
			return 0.65 // sprint away when startled by a close rival
		}
	case PersonalityAggressive:
		if action >= 7 && nearRivalDist < 0.5 && canBoost {
			return 0.75 // boost into cutoffs when a rival is in range
		}
	case PersonalityGreedy:
		if action >= 7 && obs[newest+foodBlock+3] > 0.35 && canBoost {
			return 0.55 // sprint for rich corpse piles
		}
	}
	return 0
}

// SetPolicy copies a validated policy before publishing it to the world.
func (w *World) SetPolicy(p *Policy) error {
	if p != nil {
		if err := p.Validate(); err != nil {
			return err
		}
		data, _ := json.Marshal(p)
		var c Policy
		_ = json.Unmarshal(data, &c)
		p = &c
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.policy = p
	w.policyStacks = nil
	return nil
}
func clamp(v float64) float64 { return math.Max(-1, math.Min(1, v)) }
func (w *World) applyAction(s *Snake, action, frames int) {
	s.TargetHeading = s.Heading + actionTurns[action%7]
	s.BoostUntil = w.tick
	if action >= 7 && s.Score > MinScore {
		s.BoostUntil = w.tick + uint64(frames) + 1
	}
}

// frameStack keeps the last FrameCount single frames, oldest first.
// The first push fills every slot, so a fresh episode has no zero frames.
type frameStack struct {
	frames [][]float64
}

func (f *frameStack) reset() { f.frames = nil }

func (f *frameStack) push(single []float64) []float64 {
	frame := append([]float64(nil), single...)
	if len(f.frames) == 0 {
		f.frames = make([][]float64, FrameCount)
		for i := range f.frames {
			f.frames[i] = append([]float64(nil), frame...)
		}
	} else {
		copy(f.frames, f.frames[1:])
		f.frames[FrameCount-1] = frame
	}
	return f.peek()
}

func (f *frameStack) peek() []float64 {
	out := make([]float64, ObservationSize)
	if len(f.frames) == 0 {
		return out
	}
	for i, frame := range f.frames {
		copy(out[i*SingleObservationSize:], frame)
	}
	return out
}

// policyObservation returns the input the live policy consumes: the
// legacy 49-vector for v1 policies, the stacked v2 vector otherwise.
func (w *World) policyObservation(s *Snake) []float64 {
	if w.policy != nil && w.policy.Version == LegacyPolicyVersion {
		return w.observeLegacy(s)
	}
	if w.policyStacks == nil {
		w.policyStacks = map[int]*frameStack{}
	}
	stack := w.policyStacks[s.ID]
	if stack == nil {
		stack = &frameStack{}
		w.policyStacks[s.ID] = stack
	}
	if !s.Alive {
		stack.reset()
	}
	return stack.push(w.observeSingle(s))
}

// observe is the current (stacked v2) observation for a standalone snake
// with no history; it fills the stack with the current frame. Training
// callers that step repeatedly own a frameStack instead.
func (w *World) observe(s *Snake) []float64 {
	var stack frameStack
	return stack.push(w.observeSingle(s))
}

func speedOf(s *Snake) float64 {
	if s.speed > 0 {
		return s.speed
	}
	return 0.8
}
func boostingOf(s *Snake, tick uint64) float64 {
	if s.boosting || ((s.MouseBoost || s.BoostUntil > tick) && s.Score > MinScore) {
		if s.Score > MinScore {
			return 1
		}
	}
	return 0
}

// observeSingle builds one v2 frame; see the layout comment above.
func (w *World) observeSingle(s *Snake) []float64 {
	o := make([]float64, SingleObservationSize)
	h := s.Segments[0]
	f := direction(s.Heading)
	side := Vec2{-f.Y, f.X}
	local := func(v Vec2) (float64, float64) { return v.X*f.X + v.Y*f.Y, v.X*side.X + v.Y*side.Y }
	// Self block.
	o[0] = clamp(s.Score / 100)
	o[1] = clamp(math.Log(math.Max(1, s.Score/MinScore)) / 3)
	if s.Score > MinScore {
		o[2] = 1
	}
	o[3] = clamp((w.radius - h.Len()) / 40)
	if d := h.Len(); d > 0 {
		x, y := local(h.Scale(-1 / d))
		radial := clamp(d / w.radius)
		pull := math.Max(0, math.Min(1, (radial-0.60)/0.40))
		o[4], o[5] = x*pull, y*pull
		o[15] = radial
	}
	o[6] = clamp(speedOf(s) / 1.6)
	o[7] = boostingOf(s, w.tick)
	o[8] = clamp(s.turnRate)
	o[9] = clamp(s.prevTurn)
	o[10] = clamp(s.prevBoost)
	o[11] = clamp(float64(len(s.Segments)) / 120)
	o[13] = s.Score - math.Floor(s.Score)
	o[14] = clamp((s.Score - MinScore) / 50)
	// Food targets, best first.
	for i, t := range w.foodTargets(s, foodTargetsCount) {
		x, y := local(t.pos.Sub(h))
		base := foodBlock + i*4
		o[base] = clamp(x / 40)
		o[base+1] = clamp(y / 40)
		o[base+2] = clamp(t.dist / 60)
		o[base+3] = clamp(t.value / 18)
	}
	// Rivals, nearest first with a stable ID tie-break.
	type rival struct {
		other *Snake
		delta Vec2
		dist  float64
	}
	rivals := []rival{}
	ids := w.snakeIDs()
	for _, id := range ids {
		other := w.snakes[id]
		if !other.Alive || other.ID == s.ID || len(other.Segments) == 0 {
			continue
		}
		delta := other.Segments[0].Sub(h)
		rivals = append(rivals, rival{other, delta, delta.Len()})
	}
	sort.Slice(rivals, func(i, j int) bool {
		if rivals[i].dist == rivals[j].dist {
			return rivals[i].other.ID < rivals[j].other.ID
		}
		return rivals[i].dist < rivals[j].dist
	})
	if len(rivals) > 0 {
		o[12] = math.Min(1, rivals[0].dist/60)
	} else {
		o[12] = 1
	}
	for i := 0; i < rivalCount && i < len(rivals); i++ {
		r := rivals[i]
		base := rivalBlock + i*rivalStride
		x, y := local(r.delta)
		o[base] = clamp(x / 40)
		o[base+1] = clamp(y / 40)
		vel := direction(r.other.Heading).Scale(speedOf(r.other))
		vx, vy := local(vel)
		o[base+2] = clamp(vx / 1.6)
		o[base+3] = clamp(vy / 1.6)
		o[base+4] = math.Cos(r.other.Heading - s.Heading)
		o[base+5] = math.Sin(r.other.Heading - s.Heading)
		o[base+6] = math.Min(1, r.other.Score/(2*s.Score))
		o[base+7] = math.Min(1, r.dist/60)
		o[base+8] = boostingOf(r.other, w.tick)
		o[base+9] = clamp(r.other.turnRate)
		o[base+10] = 1
	}
	// Sectors: precompute segment deltas once, then cast 24 rays.
	type seg struct {
		delta Vec2
		head  bool
	}
	segs := []seg{}
	for _, id := range ids {
		other := w.snakes[id]
		if !other.Alive || other.ID == s.ID {
			continue
		}
		for j, p := range other.Segments {
			segs = append(segs, seg{p.Sub(h), j == 0})
		}
	}
	for ray := 0; ray < sectorCount; ray++ {
		a := float64(ray) * 2 * math.Pi / sectorCount
		dir := direction(s.Heading + a)
		bodyClear, headClear := 30.0, 30.0
		for _, sg := range segs {
			along := sg.delta.X*dir.X + sg.delta.Y*dir.Y
			across := math.Abs(sg.delta.X*dir.Y - sg.delta.Y*dir.X)
			if along >= 0 && across < 1.6 {
				v := math.Max(0, along-0.85)
				if sg.head {
					headClear = math.Min(headClear, v)
				} else {
					bodyClear = math.Min(bodyClear, v)
				}
			}
		}
		dot := h.X*dir.X + h.Y*dir.Y
		border := -dot + math.Sqrt(math.Max(0, dot*dot+w.radius*w.radius-h.Len()*h.Len()))
		base := sectorBlock + ray*sectorStride
		o[base] = bodyClear / 30
		o[base+1] = headClear / 30
		o[base+2] = math.Min(1, border/40)
	}
	for _, p := range w.pellets {
		delta := p.Pos.Sub(h)
		d := delta.Len()
		if d > 40 {
			continue
		}
		a := math.Atan2(delta.Y, delta.X) - s.Heading
		sector := int(math.Round(a / (2 * math.Pi / sectorCount)))
		sector = (sector%sectorCount + sectorCount) % sectorCount
		o[sectorBlock+sector*sectorStride+3] += p.Value / (1 + d) * 0.3
	}
	for ray := 0; ray < sectorCount; ray++ {
		i := sectorBlock + ray*sectorStride + 3
		o[i] = math.Min(1, o[i])
	}
	return o
}

// observeLegacy is the v1 49-feature observation, kept so v1 policies
// and their validation history remain comparable.
func (w *World) observeLegacy(s *Snake) []float64 {
	o := make([]float64, LegacyObservationSize)
	h := s.Segments[0]
	f := direction(s.Heading)
	side := Vec2{-f.Y, f.X}
	local := func(v Vec2) (float64, float64) { return v.X*f.X + v.Y*f.Y, v.X*side.X + v.Y*side.Y }
	o[0] = clamp(s.Score / 100)
	if s.Score > MinScore {
		o[1] = 1
	}
	o[2] = clamp((w.radius - h.Len()) / 40)
	if d := h.Len(); d > 0 {
		x, y := local(h.Scale(-1 / d))
		o[3], o[4] = x, y
	}
	food := w.foodGoal(s)
	x, y := local(food.Sub(h))
	o[5], o[6] = clamp(x/40), clamp(y/40)
	nearest := math.Inf(1)
	ids := w.snakeIDs()
	for _, id := range ids {
		other := w.snakes[id]
		if !other.Alive || other.ID == s.ID {
			continue
		}
		delta := other.Segments[0].Sub(h)
		if d := delta.Len(); d < nearest {
			nearest = d
			x, y = local(delta)
			o[7], o[8] = clamp(x/40), clamp(y/40)
			o[9] = math.Cos(other.Heading - s.Heading)
			o[10] = math.Sin(other.Heading - s.Heading)
			o[11] = math.Min(1, other.Score/(2*s.Score))
		}
	}
	o[12] = math.Min(1, nearest/60)
	for ray := 0; ray < 12; ray++ {
		a := float64(ray) * 2 * math.Pi / 12
		dir := direction(s.Heading + a)
		clearance := 30.0
		for _, id := range ids {
			other := w.snakes[id]
			if !other.Alive || other.ID == s.ID {
				continue
			}
			for _, p := range other.Segments {
				d := p.Sub(h)
				along := d.X*dir.X + d.Y*dir.Y
				across := math.Abs(d.X*dir.Y - d.Y*dir.X)
				if along >= 0 && across < 1.6 {
					clearance = math.Min(clearance, math.Max(0, along-0.85))
				}
			}
		}
		dot := h.X*dir.X + h.Y*dir.Y
		border := -dot + math.Sqrt(math.Max(0, dot*dot+w.radius*w.radius-h.Len()*h.Len()))
		o[13+ray*3] = clearance / 30
		o[14+ray*3] = math.Min(1, border/40)
	}
	for _, p := range w.pellets {
		delta := p.Pos.Sub(h)
		d := delta.Len()
		if d > 40 {
			continue
		}
		a := math.Atan2(delta.Y, delta.X) - s.Heading
		sector := int(math.Round(a / (2 * math.Pi / 12)))
		sector = (sector%12 + 12) % 12
		o[15+sector*3] += p.Value / (1 + d) * 0.3
	}
	for ray := 0; ray < 12; ray++ {
		i := 15 + ray*3
		o[i] = math.Min(1, o[i])
	}
	return o
}
