# DRL Final Project

## Available Command 
```
python scripts/DQN_train.py 
```

```
python scripts/train.py --algo DQN --exp experiment_1 --debug False
```

## Tree
```
.
├── Game_2048
│   ├── board.py
│   └── __pycache__
│       └── board.cpython-310.pyc
├── params
│   ├── DoubleDQN.yaml
│   ├── DQN.yaml
│   └── DuelingDQN.yaml
├── readme.md
├── result
│   ├── DoubleDQN
│   │   ├── experiment_1
│   │   │   └── log_abc.csv
│   │   └── experiment_2
│   │       └── log_abc.csv
│   ├── DQN
│   │   ├── experiment_1
│   │   │   └── log_abc.csv
│   │   └── experiment_2
│   │       └── log_abc.csv
│   └── DuelingDQN
│       ├── experiment_1
│       │   └── log_abc.csv
│       └── experiment_2
│           └── log_abc.csv
├── RL_Algorithm
│   ├── Algorithm
│   │   ├── DoubleDQN.py
│   │   ├── DQN.py
│   │   ├── DuelingDQN.py
│   │   └── __pycache__
│   │       └── DeepQLearning.cpython-310.pyc
│   └── RL_base.py
├── scripts
│   ├── DQN_train.py
│   ├── manual.py
│   ├── play.py
│   ├── __pycache__
│   │   ├── manual.cpython-310.pyc
│   │   └── random_action.cpython-310.pyc
│   ├── random_action.py
│   └── train.py
└── weight
    ├── DoubleDQN
    │   ├── experiment_1
    │   │   ├── policy_net.pth
    │   │   └── target_net.pth
    │   └── experiment_2
    │       ├── policy_net.pth
    │       └── target_net.pth
    ├── DQN
    │   ├── experiment_1
    │   │   ├── policy_net.pth
    │   │   └── target_net.pth
    │   └── experiment_2
    │       ├── policy_net.pth
    │       └── target_net.pth
    └── DuelingDQN
        ├── experiment_1
        │   ├── policy_net.pth
        │   └── target_net.pth
        └── experiment_2
            ├── policy_net.pth
            └── target_net.pth
```

### Ideal Commands

#### Train
```
python scripts/train.py --algo DQN --exp exp1 --debug False
```

#### Test 
```
python scripts/train.py --algo DQN --exp exp1 --debug True
```


## reference
- https://github.com/qwert12500/2048_rl
