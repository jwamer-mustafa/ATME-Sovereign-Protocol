package main

type Config struct {
	Pool     string
	Port     string
	User     string
	Password string
	LogFile  string
}

func LoadConfig() Config {
	return Config{
		Pool:     "stratum+tcp://solo.ckpool.org",
		Port:     "3333",
		User:     "166BF3ZMzft95Cw5xjGhtBY4q2iRTdsVth",
		Password: "x",
		LogFile:  "../assets/logs/run_log.txt",
	}
}