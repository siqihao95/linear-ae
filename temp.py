# get run data from wandb
for name in model_names:
    if name == "deterministic_nd":
        model_type = ModelTypes.NESTED_DROPOUT
        optim_hdim_lrs = optimal_lrs['nd_exp']
        # extra_filters = {'nd_expectation': True}
    elif name == "nd":
        model_type = ModelTypes.NESTED_DROPOUT
        optim_hdim_lrs = optimal_lrs['nd']
        # extra_filters = {'nd_expectation': False}
    else:
        model_type = name
        optim_hdim_lrs = optimal_lrs[model_type]
        # extra_filters = {}

    for optimizer in algo_names:
        print("=> Getting data for {}".format(optimizer))
        all_runs[optimizer][name] = {}
        for hdim in hdims:
            filters = {
                'config.hdim': hdim,
                'config.train_itr': 50000,
                'config.lr': optim_hdim_lrs[optimizer][hdim],
                # **extra_filters
            }
            runs = api.runs('sheldonhuang/lae-rms-naive', filters=filters)

            # filtered_runs = runs
            all_runs = runs
            # if len(filtered_runs) == 1:
            #     all_runs[optimizer][name][hdim] = filtered_runs[0]
            # else:
            #     print("Warning: {} runs found with {}, {}, hdim {}".format(
            #         len(filtered_runs), optimizer, name, hdim))
            #     for run in filtered_runs:
            #         if "paper" in run.tags:
            #             print("Accept run {} with tag 'paper'".format(run.id))
            #             all_runs[optimizer][name][hdim] = run
            #             break