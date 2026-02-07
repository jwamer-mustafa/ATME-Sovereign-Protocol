package main

import "fmt"

type PulseEngine struct {
	client *StratumClient
	logger *Logger
}

func NewPulseEngine(c *StratumClient, l *Logger) *PulseEngine {
	return &PulseEngine{c, l}
}

func (p *PulseEngine) ShouldTrigger(entropy float64) bool {
	return entropy > 0.95
}

func (p *PulseEngine) Emit() {
	fmt.Println("PULSE TRIGGERED")
	p.logger.Log("Pulse emitted")
}