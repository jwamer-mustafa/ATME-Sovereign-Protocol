package main

type Stats struct {
	Pulses int
}

func (s *Stats) RecordPulse() {
	s.Pulses++
}