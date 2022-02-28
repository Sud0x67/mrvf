# pymarl 调用过程
function call stack
main --> run --> [runner ( --> mac 收集样本) + learner( --> mac forward 策略)]
# main func
## 配置优先级 alg > env > default (default 已经删除)
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    # run
    run_REGISTRY[_config['run']](_run, config, _log)
# run 
 - run
    - run()-> run_sequential()->
                -runner
                    - runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
                -mac
                    - mac_REGISTRY[args.mac](buffer.scheme, groups, args)
                -buffer
                    - scheme
                -learner
                    - (mac, buffer.scheme, logger, args)
# runner learner and mac
    runner and learner both handle the mac, but learner learn experience while runner try any trace.
    runner.run() and learner.train() is the API provoked by run.
# runner
    eisode runner run one episode when runner.run() is provoked.
    and Parallel_runner run args.episode_run times when runner.run is provoked.
#  MAC
    N_controller --> Basic Controller
# buffer scheme
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
here vshape means the shape of variety and group means its the property of group. dtype means the type of data.

# logger
# nq_learner
    nq_learner 构造一个tderror learn 是对一个序列学习的。
# run
    log：
    save：
    test：
# Mixer
### !!!!
    qmix.py is to refined, the nmix.py is used in fact.
# results
    关于result, result 中1个文件夹代表一次实验的结果，info.json 保存了所有的必要信息：
    _log 代表sacred的_log类， Logger 托管了 _log " main(_log) -> run(Logger(_log)) ->runner(logger)
# last about the config files.
    alg_config > env_config > defaul_config
# 保存 replay
- 使用episode runner 保存回放， parrallel 没有实现
- 更改以下几处设置
    - replay_dir: replay_prefix in `config/envs/sc2.ymal` ***Attention: replay_dir: must be absolute path***
      - replay_dir: 不用更改
    - runner: "episode" batch_size_run: 1 in `config/algs/xx.ymal`
    - or save_replay: true checkpoint_path: in `default.ymal` checkpoint_path 默认应该为空
- 提前save model
    - evaluate: False and save_replay: False in `default.ymal`
    - runner: 'parallel' batch_size_run: 8 in `config/algs/xx.ymal`
    - save_model_interval: 50000, save_model: True in `default.ymal`
- replay主机设置
    - 安装 starcraftII
    - 将所用版本SC-II的 Battel.net/ Libs/ Versions/ 复制到 SC-II的安装目录（理论上可只复制Battel.net/）
    - 直接打开replay文件
    


