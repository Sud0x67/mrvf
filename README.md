# The Code for Our Paper MRVF. 



Here is the code for our paper ***"Priority Over Quantity: A Self-Incentive Credit Assignment Scheme for Cooperative Multiagent Reinforcement Learning"***. 

Cite with:
```
@article{DBLP:journals/tcss/TangWCZ24,
  author       = {Hao Tang and
                  Cheng Wang and
                  Shengbo Chang and
                  Junqi Zhang},
  title        = {Priority Over Quantity: {A} Self-Incentive Credit Assignment Scheme
                  for Cooperative Multiagent Reinforcement Learning},
  journal      = {{IEEE} Trans. Comput. Soc. Syst.},
  volume       = {11},
  number       = {6},
  pages        = {7766--7777},
  year         = {2024},
  url          = {https://doi.org/10.1109/TCSS.2024.3428334},
  doi          = {10.1109/TCSS.2024.3428334},
  timestamp    = {Wed, 08 Jan 2025 21:12:03 +0100},
  biburl       = {https://dblp.org/rec/journals/tcss/TangWCZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

You can run the experiments by the command below:
```
python src/main.py --config=mrmix --env-config=sc2 with env_args.map_name=3m
```
For the dependencies and SMAC benchmark, you can refer to [pymarl2](https://github.com/hijkzzz/pymarl2).
# Notice
We use "res_qmix" for the name of our method at early stage. You can also run the experiments by
```
python src/main.py --config=res_qmix_*** --env-config=sc2 with env_args.map_name=3m
```
. It works identically as the former command.