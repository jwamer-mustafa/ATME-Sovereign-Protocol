package main

import (
	"fmt"
	"time"
)

func main() {

	fmt.Println("ATME Core Starting...")

	cfg := LoadConfig()

	client := NewStratumClient(cfg)
	go client.Connect()

	logger := NewLogger(cfg.LogFile)

	pulse := NewPulseEngine(client, logger)

	entropy := NewEntropyEngine()

	for {
		e := entropy.Sample()

		if pulse.ShouldTrigger(e) {
			pulse.Emit()
		}

		time.Sleep(20 * time.Millisecond)
	}
}