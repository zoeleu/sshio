package game

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

// twoLayerPolicy routes input 0 through two tanh units so the second layer
// matters: logit[4] = tanh(1*tanh(2*x)) > bias action 3 only when x > 0.
func twoLayerPolicy() *Policy {
	p := &Policy{
		Version: PolicyVersion,
		W1:      [][]float64{make([]float64, ObservationSize)},
		B1:      []float64{0},
		WH:      [][]float64{{1}},
		BH:      []float64{-0.5},
		W2:      make([][]float64, ActionCount),
		B2:      make([]float64, ActionCount),
	}
	for i := range p.W2 {
		p.W2[i] = []float64{0}
	}
	p.W1[0][0] = 2
	p.W2[4][0] = 1
	p.B2[3] = 0.1
	return p
}

func TestTwoHiddenLayerInference(t *testing.T) {
	p := twoLayerPolicy()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	o := make([]float64, ObservationSize)
	// x=0: h1=0, h2=tanh(-0.5)<0, logit4<0 < bias 0.1 -> action 3.
	if got := p.Action(o); got != 3 {
		t.Fatalf("expected bias action 3, got %d", got)
	}
	// x=1: h1=tanh(2)~0.964, h2=tanh(0.964-0.5)~0.43 > 0.1 -> action 4.
	o[0] = 1
	if got := p.Action(o); got != 4 {
		t.Fatalf("expected action 4, got %d", got)
	}
	// The bias on layer 2 must actually be applied: without it h2~0.75.
	p.BH[0] = -2
	if got := p.Action(o); got != 3 {
		t.Fatalf("second-layer bias ignored, got action %d", got)
	}
}

func TestTwoHiddenLayerValidation(t *testing.T) {
	bad := func(name string, mutate func(*Policy)) {
		p := twoLayerPolicy()
		mutate(p)
		if p.Validate() == nil {
			t.Fatalf("accepted %s", name)
		}
	}
	bad("bias/weight length mismatch", func(p *Policy) { p.BH = nil })
	bad("wrong hidden fan-in", func(p *Policy) { p.WH[0] = []float64{1, 1} })
	bad("wrong output fan-in", func(p *Policy) { p.W2[0] = []float64{0, 0} })
	bad("NaN in second layer", func(p *Policy) { p.WH[0][0] = math.NaN() })
	bad("Inf in second bias", func(p *Policy) { p.BH[0] = math.Inf(1) })
	bad("oversized second layer", func(p *Policy) {
		p.WH = make([][]float64, maxHiddenWidth+1)
		p.BH = make([]float64, maxHiddenWidth+1)
	})
}

func TestTwoHiddenLayerJSONRoundTripAndLegacyLoad(t *testing.T) {
	data, err := json.Marshal(twoLayerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	var p Policy
	if err = json.Unmarshal(data, &p); err != nil || p.Validate() != nil || len(p.WH) != 1 {
		t.Fatalf("round trip failed: %v %v", err, p.Validate())
	}
	// A single-layer export must not gain second-layer keys.
	legacy, _ := json.Marshal(testPolicy())
	var keys map[string]json.RawMessage
	_ = json.Unmarshal(legacy, &keys)
	if _, ok := keys["wh"]; ok {
		t.Fatal("single-layer policy serialized wh")
	}
}

func randomPolicy(h1, h2 int) *Policy {
	r := rand.New(rand.NewSource(1))
	mat := func(rows, cols int) [][]float64 {
		m := make([][]float64, rows)
		for i := range m {
			m[i] = make([]float64, cols)
			for j := range m[i] {
				m[i][j] = r.NormFloat64() * 0.05
			}
		}
		return m
	}
	p := &Policy{Version: PersonalityPolicyVersion, W1: mat(h1, PersonalityObservationSize),
		B1: make([]float64, h1), B2: make([]float64, ActionCount)}
	last := h1
	if h2 > 0 {
		p.WH, p.BH, last = mat(h2, h1), make([]float64, h2), h2
	}
	p.W2 = mat(ActionCount, last)
	return p
}

func benchmarkPolicy(b *testing.B, p *Policy) {
	obs := make([]float64, PersonalityObservationSize)
	for i := range obs {
		obs[i] = rand.Float64()*2 - 1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Action(obs)
	}
}

func BenchmarkPolicy64(b *testing.B)      { benchmarkPolicy(b, randomPolicy(64, 0)) }
func BenchmarkPolicy256(b *testing.B)     { benchmarkPolicy(b, randomPolicy(256, 0)) }
func BenchmarkPolicy256x128(b *testing.B) { benchmarkPolicy(b, randomPolicy(256, 128)) }
