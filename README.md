# The Code for Our Paper MrMIX. 
Here is the code for our paper "MrMIX: Multihead Residual Value Factorization for Cooperative Multi-Agent Reinforcement Learning". You can run the experiments by the command below:
```
python src/main.py --config=mrmix --env-config=sc2 with env_args.map_name=3m
```
For the dependencies and SMAC benchmark, you can refer to [pymarl2](https://github.com/hijkzzz/pymarl2).
# Notice
We use "res_qmix" for the name our method at early stage. You can also run the experiments by
```
python src/main.py --config=res_qmix_*** --env-config=sc2 with env_args.map_name=3m
```
. It works identically as the above command.

We havn't update the name in codes and filename for the reason of time, which will be updated later. 
