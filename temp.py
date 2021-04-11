def draw_scatterplot(cutoff, all_hdims):
    all_curves = {
        "RMSprop_full": {
            "hdim": [],
            "smallest_epoch": []
        },
        "SGD": {
            "hdim": [],
            "smallest_epoch": []
        },
        "RMSprop_subspace_only": {
            "hdim": [],
            "smallest_epoch": []
        }
    }

    for optimizer in algo_names:
        for hdim in hdims:
            #             for name in model_names:
            if hdim not in histories[optimizer][name].keys():
                continue
            history_i = histories[optimizer][hdim]  #.fillna(1.0)
            #                 model_type = all_runs[optimizer][name][hdim].config['model_type']

            cutoff_indices = history_i[
                history_i['axis-alignment/rotation'].astype(
                    np.float).le(cutoff)].index
            if len(cutoff_indices) > 0:
                all_curves[optimizer]['hdim'].append(hdim)
                all_curves[optimizer]['smallest_epoch'].append(
                    history_i['_step'][
                        cutoff_indices[0]])  # first index below cutoff
    colors = {
        "RMSprop_full": 'tab:blue',
        "SGD": 'tab:green',
        "RMSprop_subspace_only": 'tab:red',
        #         'deterministic_nd': 'tab:orange',
        #         ModelTypes.VAE: 'tab:purple'
    }

    mark_shapes = {
        'RMSprop_full': '-o',
        'SGD': '-o',
        'RMSprop_subspace_only': '-o',
        #         'Adam': '--o'
    }
    end_shapes = {
        #         'SGD': '-',
        'RMSprop_full': '-',
        'SGD': '-',
        'RMSprop_subspace_only': '-',
        #         'Adam': '--'
    }

    with seaborn.axes_style('darkgrid'):
        for optimizer in algo_names:
            for name in list(colors.keys()):
                epochs = []

                curves_hdims = all_curves[optimizer]
                if optimizer == 'SGD':
                    plt.plot(curves_hdims['hdim'][:-1],
                             curves_hdims['smallest_epoch'][:-1],
                             mark_shapes[optimizer],
                             label=name,
                             color=colors[name])
                else:
                    # no label
                    plt.plot(curves_hdims['hdim'][:-1],
                             curves_hdims['smallest_epoch'][:-1],
                             mark_shapes[optimizer],
                             color=colors[name])
                plt.plot(curves_hdims['hdim'][-2:],
                         curves_hdims['smallest_epoch'][-2:],
                         end_shapes[optimizer],
                         color=colors[name])

                if len(curves_hdims['hdim']) == 0:
                    continue
                last_hdim = curves_hdims['hdim'][-1]
                last_epoch = curves_hdims['smallest_epoch'][-1]

                if all_hdims[-1] not in curves_hdims['hdim']:
                    last_hdim = curves_hdims['hdim'][-1]
                    last_epoch = curves_hdims['smallest_epoch'][-1]
                    plt.scatter([last_hdim], [last_epoch],
                                s=50,
                                linewidth=3,
                                marker='x',
                                color=colors[name])

                else:
                    plt.scatter([last_hdim], [last_epoch],
                                marker='o',
                                color=colors[name])
        plt.legend(loc='lower right')
        plt.ylim(10, 200000)
        plt.xlim(1.5, all_hdims[-1] * 1.5)
        plt.xlabel("Hidden dimension")
        plt.ylabel('Epoch')
        plt.yscale('log')
        plt.xscale('log')
        #         plt.title('Epochs to reach {} axis-alignment distance'.format(cutoff))
        plt.savefig('plots/paper/fig4_{}.pdf'.format(cutoff),
                    bbox_inches='tight')
