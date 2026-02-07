package main

import (
	"fmt"
	"os"
	"time"
)

type Logger struct {
	file *os.File
}

func NewLogger(path string) *Logger {

	file, _ := os.Create(path)

	return &Logger{file: file}
}

func (l *Logger) Log(msg string) {
	timestamp := time.Now().Format(time.RFC3339)
	line := fmt.Sprintf("[%s] %s\n", timestamp, msg)
	l.file.WriteString(line)
}