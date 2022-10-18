import numpy
import seaborn as sns
import pandas as pd

from . import plottools as pt


def diversity(checkpoint, evaluator, color='b', figs={}, reportname=''):
    '''plot the whole history, the hall of fame, and the best individual
    from a unumpyickled checkpoint
    '''

    param_names = evaluator.param_names
    n_params = len(param_names)
    print(f"n_params: {n_params}")

    hof = checkpoint['halloffame']
    fitness_cut_off = 2.*sum(hof[0].fitness.values)
    print(f"fitness_cut_off: {fitness_cut_off}")

    figname = 'Diversity ' + reportname
    fig = pt.make_figure(figname=figname,
                        orientation='page',
                        figs=figs,
                        fontsizes = (10,10))

    axs = pt.tiled_axs(frames=n_params, columns=6, d_out=5, fig=fig,
                top_margin=0.08, bottom_margin=0.03,
                left_margin=0.08, right_margin=0.03,
                hspace=0.3, wspace=0.8)

    all_params = get_params(checkpoint['history'].genealogy_history.values(),
                            fitness_cut_off=fitness_cut_off)
    hof_params = get_params(checkpoint['halloffame'],
                            fitness_cut_off=fitness_cut_off)

    best_params = checkpoint['halloffame'][0]

    for i, name in enumerate(param_names):

        ax = axs[i]

        p1 = numpy.array(all_params)[:, i].tolist()
        p2 = numpy.array(hof_params)[:, i].tolist()

        df = pd.DataFrame()
        df['val'] = p1+p2
        df['type'] = ['all']*len(p1)+['hof']*len(p2)
        df['param'] = [name]*(len(p1)+len(p2))

        sns.violinplot(x='param', y='val', hue='type', data=df, ax=ax, split=True,
               inner="quartile", palette={"all": "gray", "hof": color})

        ax.axhline(y=best_params[i], color=color, label='best', linewidth=2)
        ax.set_ylabel('')

        if i > 0:
            ax.legend_.remove()
        else:
            ax.legend()


    for i, parameter in enumerate(evaluator.params):
        min_value = parameter.lower_bound
        max_value = parameter.upper_bound

        ax = axs[i]
        ax.set_ylim((min_value-0.02*min_value, max_value+0.02*max_value))

        name = param_names[i]
        label = name.replace('.', '\n')

        ax.set_title(label, fontsize=7)

        pt.adjust_spines(ax, ['left'], d_out=5)

        #ax.set_xticks([])
        #ax.axes.get_xaxis().set_visible(False)


    fig.suptitle(reportname, fontsize=10)

    return fig


def get_params(
        params,
        fitness_cut_off=1e9
        ):

    '''plot the individual parameter values'''
    results = []
    for param in params:
        if fitness_cut_off > sum(param.fitness.values):
            results.append(param)

    return results
