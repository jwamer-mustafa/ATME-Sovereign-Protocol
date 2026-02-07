package main

import (
	"bufio"
	"fmt"
	"net"
)

type StratumClient struct {
	config Config
	conn   net.Conn
}

func NewStratumClient(cfg Config) *StratumClient {
	return &StratumClient{config: cfg}
}

func (s *StratumClient) Connect() {

	address := s.config.Pool + ":" + s.config.Port

	conn, err := net.Dial("tcp", address)
	if err != nil {
		fmt.Println("Connection failed:", err)
		return
	}

	s.conn = conn

	fmt.Println("Connected to pool")

	s.subscribe()
	s.authorize()

	s.listen()
}

func (s *StratumClient) subscribe() {
	msg := `{"id":1,"method":"mining.subscribe","params":[]}` + "\n"
	s.conn.Write([]byte(msg))
}

func (s *StratumClient) authorize() {
	msg := fmt.Sprintf(
		`{"id":2,"method":"mining.authorize","params":["%s","%s"]}`+"\n",
		s.config.User,
		s.config.Password,
	)
	s.conn.Write([]byte(msg))
}

func (s *StratumClient) listen() {

	reader := bufio.NewReader(s.conn)

	for {
		line, _ := reader.ReadString('\n')
		fmt.Println("POOL:", line)
	}
}