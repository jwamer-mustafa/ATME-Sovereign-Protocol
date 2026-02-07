package main

import (
	"math/rand"
	"time"
)

type EntropyEngine struct{}

func NewEntropyEngine() *EntropyEngine {
	rand.Seed(time.Now().UnixNano())
	return &EntropyEngine{}
}

func (e *EntropyEngine) Sample() float64 {
	return rand.Float64()
}