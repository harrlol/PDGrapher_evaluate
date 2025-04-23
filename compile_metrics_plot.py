import re
import glob
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def extract_results_from_txt(path):
    with open(path, 'r') as f:
        content = f.read()

    # Use regex to find the list of dicts
    matches = re.findall(r'\[\{.*?\}\]', content, re.DOTALL)
    results = []
    for m in matches:
        try:
            results.append(json.loads(m.replace("'", '"')))  # Fix single quotes to valid JSON
        except json.JSONDecodeError:
            continue
    return results

# note cell lines have to be in the right order!!
def collect_metrics(folder_path, curr_cell_line):
    all_results = []
    for filepath in glob.glob(os.path.join(folder_path, "*txt")):
        info = os.path.basename(filepath).replace(".txt", "").split("_")
        model = info[0]
        pert_mode = info[1]
        metric_raw = extract_results_from_txt(filepath)
        for i in range(len(metric_raw)):
            cell_line = curr_cell_line[i]
            for rep_idx, run in enumerate(metric_raw[i]):
                entry = {
                    "model": model,
                    "perturbation": pert_mode,
                    "cell_line": cell_line,
                    "split": rep_idx + 1
                }
                for split in ["train", "test"]:
                    for metric_name, val in run[split].items():
                        entry[f"{split}_{metric_name}"] = val
                all_results.append(entry)

    df = pd.DataFrame(all_results)
    return df


def process_metrics(df):
    df_avg = df.groupby(["cell_line", "model", "perturbation"]).mean(numeric_only=True).reset_index().drop(columns=["split"])
    df_std = df.groupby(["cell_line", "model", "perturbation"]).std().reset_index().drop(columns=["split"])
    df_std = df_std.add_suffix('_std')
    df_std.rename(columns={
        'cell_line_std': 'cell_line',
        'model_std': 'model',
        'perturbation_std': 'perturbation'
    }, inplace=True)
    df_summary = df_avg.merge(df_std, on=["cell_line", "model", "perturbation"])
    df_summary = df_summary.round(3)
    collapsed_across_celllines = df_summary.groupby(["perturbation", "model"]).mean(numeric_only=True).reset_index()
    collapsed_across_celllines = collapsed_across_celllines.round(3)
    return df_summary, collapsed_across_celllines


def df_summary_stat(df_summary):
    df_long = df_summary.melt(
        id_vars=['cell_line', 'model', 'perturbation'],
        value_vars=[col for col in df_summary.columns if not col.endswith('_std')],
        var_name='metric',
        value_name='value'
    )

    std_cols = df_summary.melt(
        id_vars=['cell_line', 'model', 'perturbation'],
        value_vars=[col for col in df_summary.columns if col.endswith('_std')],
        var_name='metric_std',
        value_name='std'
    )

    std_cols['metric'] = std_cols['metric_std'].str.replace('_std$', '', regex=True)

    df_long = df_long.merge(std_cols[['cell_line', 'model', 'perturbation', 'metric', 'std']],
                            on=['cell_line', 'model', 'perturbation', 'metric'], how='left')
    
    return df_long


def plot_metric(df_long, direction, split, folder_path):

    base_metrics = ['r2_scgen', 'spearman', 'r2']
    plot_metrics = [f'{split}_{direction}_{m}' for m in base_metrics]
    plot_data = df_long[df_long['metric'].isin(plot_metrics)]

    g = sns.catplot(
        data=plot_data,
        x='cell_line',
        y='value',
        hue='model',
        col='metric',
        row='perturbation',
        kind='bar',
        height=4,
        aspect=1,
        ci=None
    )

    # Add error bars + value labels
    for ax, m in zip(g.axes.flat, plot_metrics * len(plot_data['perturbation'].unique())):
        subset = plot_data[plot_data['metric'] == m].reset_index()
        for idx, bar in enumerate(ax.patches):
            if idx >= len(subset): break
            bar_height = bar.get_height()
            std = subset.iloc[idx]['std']
            ax.errorbar(
                bar.get_x() + bar.get_width()/2,
                bar_height,
                yerr=std,
                fmt='none',
                c='black',
                capsize=2
            )
            # Add text label
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar_height + std + 0.01,  # space above error bar
                f"{bar_height:.2f}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=0
            )

    # Save the plot
    g.savefig(os.path.join(folder_path, f"{direction}_{split}_metrics.pdf"), bbox_inches='tight')

    # Plot backward_avg_topk if needed
    if direction == 'backward':
        plot_metrics = [f'{split}_backward_avg_topk']
        plot_data = df_long[df_long['metric'].isin(plot_metrics)]

        g2 = sns.catplot(
            data=plot_data,
            x='cell_line',
            y='value',
            hue='model',
            col='metric',
            row='perturbation',
            kind='bar',
            height=4,
            aspect=1,
            ci=None
        )

        for ax, m in zip(g2.axes.flat, plot_metrics * len(plot_data['perturbation'].unique())):
            subset = plot_data[plot_data['metric'] == m].reset_index()
            for idx, bar in enumerate(ax.patches):
                if idx >= len(subset): break
                bar_height = bar.get_height()
                std = subset.iloc[idx]['std']
                ax.errorbar(
                    bar.get_x() + bar.get_width()/2,
                    bar_height,
                    yerr=std,
                    fmt='none',
                    c='black',
                    capsize=2
                )
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar_height + std + 10,  # adjust for avg_topk scale
                    f"{bar_height:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=0
                )

        # Save the plot
        g2.savefig(os.path.join(folder_path, f"{direction}_{split}_avg_topk_metrics.pdf"), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot metrics from training results.")
    parser.add_argument('--project_directory', type=str, required=True, help="Path to the directory of results.")
    parser.add_argument('--cell_lines', type=str, required=True, help="A string of the cell lines used in this run.")
    
    args = parser.parse_args()
    cell_lines = args.cell_lines.split()
    project_dir = args.project_directory

    out = collect_metrics(project_dir, cell_lines)
    df_summary, collapsed_across_celllines = process_metrics(out)

    # save some tables
    out.to_csv(os.path.join(project_dir, "raw_metrics.csv"), index=False)
    df_summary.to_csv(os.path.join(project_dir, "collapsed_across_splits.csv"), index=False)
    collapsed_across_celllines.to_csv(os.path.join(project_dir, "collapsed_across_celllines.csv"), index=False)

    # plot
    df_long = df_summary_stat(df_summary)
    plot_metric(df_long, "forward", "train", project_dir)
    plot_metric(df_long, "backward", "train", project_dir)
    plot_metric(df_long, "forward", "test", project_dir)
    plot_metric(df_long, "backward", "test", project_dir)

if __name__ == "__main__":
    main()