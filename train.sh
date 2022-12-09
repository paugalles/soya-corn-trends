#!/bin/bash

# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=1000
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=50
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80
# python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80
# python3 train.py --n_epochs=2000 --learning_rate=0.1 --batch_sz=80
# python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80
# python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=80
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=20
# python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=20
# python3 train.py --n_epochs=4000 --learning_rate=0.1 --batch_sz=80
# python3 train.py --n_epochs=4000 --learning_rate=0.01 --batch_sz=20


# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=50 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=2000 --learning_rate=0.1 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=20 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=20 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=4000 --learning_rate=0.1 --batch_sz=80 --weights=./model/corn.pt --instrument=CBOT.ZC
# python3 train.py --n_epochs=4000 --learning_rate=0.01 --batch_sz=20 --weights=./model/corn.pt --instrument=CBOT.ZC


python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=50 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=2000 --learning_rate=0.1 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.1 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.01 --batch_sz=20 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=1000 --learning_rate=0.001 --batch_sz=20 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=4000 --learning_rate=0.1 --batch_sz=80 --weights=./model/soya.pt --instrument=CBOT.ZS
python3 train.py --n_epochs=4000 --learning_rate=0.01 --batch_sz=20 --weights=./model/soya.pt --instrument=CBOT.ZS