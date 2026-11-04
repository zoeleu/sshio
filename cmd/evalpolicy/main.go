// evalpolicy evaluates frozen JSON policies through the authoritative Go engine.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"wriggle/game"
)

type run struct {
	Seed        int64   `json:"seed"`
	Scenario    string  `json:"scenario"`
	Return      float64 `json:"return"`
	Food        float64 `json:"food"`
	Score       float64 `json:"score"`
	Ticks       int     `json:"ticks"`
	Alive       bool    `json:"alive"`
	Kills       int     `json:"kills"`
	MutualKills int     `json:"mutual_kills"`
	Circles     int     `json:"circles"`
	BoostSpent  float64 `json:"boost_spent"`
}
type result struct {
	Policy     string  `json:"policy"`
	SHA256     string  `json:"sha256,omitempty"`
	MeanReturn float64 `json:"mean_return"`
	MeanFood   float64 `json:"mean_food"`
	MeanScore  float64 `json:"mean_score"`
	MeanTicks  float64 `json:"mean_survival_ticks"`
	Survival   float64 `json:"survival_fraction"`
	Kills      int     `json:"kills"`
	Circles    int     `json:"circles"`
	Runs       []run   `json:"runs"`
}

func main() {
	paths := flag.String("policies", "models/authored.json,heuristic,models/wriggle-v6.json", "comma-separated JSON paths or heuristic")
	seed := flag.Int64("seed", 310000, "first evaluation seed")
	episodes := flag.Int("episodes", 14, "episodes per policy; scenarios rotate in fixed order")
	horizon := flag.Int("horizon", 600, "maximum world ticks per episode")
	output := flag.String("output", "", "JSON output file (default stdout)")
	flag.Parse()
	if *episodes < 1 {
		log.Fatal("episodes must be positive")
	}
	scenarios := []string{"arena", "cutoff", "escape", "border", "forage", "chase", "production"}
	results := []result{}
	for _, path := range strings.Split(*paths, ",") {
		path = strings.TrimSpace(path)
		r := result{Policy: path}
		var p *game.Policy
		var err error
		action := -1
		if path != "heuristic" {
			p, err = game.LoadPolicy(path)
			if err != nil {
				log.Fatal(err)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				log.Fatal(err)
			}
			r.SHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
			action = -2
		}
		for i := 0; i < *episodes; i++ {
			scenario := scenarios[i%len(scenarios)]
			opponents := 3
			if scenario == "cutoff" || scenario == "escape" || scenario == "chase" {
				opponents = 1
			}
			e, err := game.NewTrainingEnv(*seed+int64(i), opponents, *horizon)
			if err != nil {
				log.Fatal(err)
			}
			if err = e.Scenario(scenario); err != nil {
				log.Fatal(err)
			}
			if err = e.SetPolicy(p); err != nil {
				log.Fatal(err)
			}
			// The teacher writes brain state; do not call it for neural evaluations.
			e.TeacherEnabled = false
			state := e.State()
			reward := 0.0
			for !state.Done {
				state, err = e.Step(action)
				if err != nil {
					log.Fatal(err)
				}
				reward += state.Reward
			}
			r.Runs = append(r.Runs, run{*seed + int64(i), scenario, reward, state.Food, state.Score,
				state.Ticks, state.Alive, state.Kills, state.MutualKills, state.Circles, state.BoostSpent})
			r.MeanReturn += reward
			r.MeanFood += state.Food
			r.MeanScore += state.Score
			r.MeanTicks += float64(state.Ticks)
			if state.Alive {
				r.Survival++
			}
			r.Kills += state.Kills
			r.Circles += state.Circles
			fmt.Fprintf(os.Stderr, "%s %d/%d %s: return=%.2f ticks=%d alive=%v\n", path, i+1, *episodes, scenario, reward, state.Ticks, state.Alive)
		}
		n := float64(*episodes)
		r.MeanReturn /= n
		r.MeanFood /= n
		r.MeanScore /= n
		r.MeanTicks /= n
		r.Survival /= n
		results = append(results, r)
	}
	report := struct {
		Suite   string   `json:"suite"`
		Horizon int      `json:"horizon"`
		Results []result `json:"results"`
	}{"Go TrainingEnv, seven scenarios, heuristic opponents, teacher disabled", *horizon, results}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	data = append(data, '\n')
	if *output == "" {
		_, err = os.Stdout.Write(data)
	} else {
		err = os.WriteFile(*output, data, 0644)
	}
	if err != nil {
		log.Fatal(err)
	}
}
