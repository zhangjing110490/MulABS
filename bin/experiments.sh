#!/bin/bash
python3 main.py --arch=FM  --log=fm  --reg=0.08  --lr=0.01
python3 main.py --arch=RF  --log=rf
python3 main.py --arch=LR  --log=lr
python3 main.py --arch=DeepFM  --log=deepfm
python3 main.py --arch=CTR  --log=ctr
python3 main.py --arch=CTR  --log=ctr  --FMembed
python3 main.py --arch=CTR  --log=ctr  --FMembed  --add_item_scores
python3 main.py --arch=CTR  --log=ctr  --add_item_scores
python3 main.py --arch=CTR  --log=ctr  --add_din
python3 main.py --arch=CTR  --log=ctr  --FMembed  --add_din