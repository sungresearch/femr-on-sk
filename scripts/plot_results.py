import os
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.default_paths import path_root
from src.mappings import model_names, task_names

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


color_palette = {
    "GBM": "dodgerblue",
    "CLMBR": "crimson",
    "CLMBR_DAPT": "#7D529D",
    "CLMBR_SK": "dodgerblue",
}

marker_style = {
    "GBM": "o",
    "CLMBR": "o",
    "CLMBR_DAPT": "s",
    "CLMBR_SK": "o",
}

model_label = {
    "GBM": "GBM",
    "CLMBR": "CLMBR",
    "CLMBR_DAPT": "CLMBR$_{DAPT}$",
    "CLMBR_SK": "CLMBR$_{SK}$",
}

tasks = [
    "In-hospital Mortality",
    "Long LOS",
    "30-day Readmission",
    "Hypoglycemia",
    "Hyponatremia",
    "Hyperkalemia",
    "Thrombocytopenia",
    "Anemia",
]


def make_table_adapter_models(
    path_to_csv: dict | str,
    models: list,
    metric: str = "AUROC",
    include_model_type: bool = False,
):
    if type(path_to_csv) == dict:
        df = (
            pd.concat(
                (pd.read_csv(v).assign(model_type=k) for k, v in path_to_csv.items())
            )
            .replace({**model_names, **task_names})
            .query("model==@models and task==@tasks")
        )

    elif type(path_to_csv) == str:
        df = (
            pd.read_csv(path_to_csv)
            .replace({**model_names, **task_names})
            .query("model==@models and task==@tasks")
        )

    df = df.assign(
        AUROC=(
            df.auroc.round(3).astype(str)
            + " ["
            + df.auroc_lower_ci.round(3).astype(str)
            + ", "
            + df.auroc_upper_ci.round(3).astype(str)
            + "]"
        ),
        AUPRC=(
            df.auprc.round(3).astype(str)
            + " ["
            + df.auprc_lower_ci.round(3).astype(str)
            + ", "
            + df.auprc_upper_ci.round(3).astype(str)
            + "]"
        ),
        AUPRC_C=(
            df.auprc_c.round(3).astype(str)
            + " ["
            + df.auprc_c_lower_ci.round(3).astype(str)
            + ", "
            + df.auprc_c_upper_ci.round(3).astype(str)
            + "]"
        ),
        ECE=(
            df.ece.round(3).astype(str)
            + " ["
            + df.ece_lower_ci.round(3).astype(str)
            + ", "
            + df.ece_upper_ci.round(3).astype(str)
            + "]"
        ),
    )

    if include_model_type:
        columns = ["model", "model_type"]
    else:
        columns = ["model"]

    return df.pivot_table(
        values=metric, index=["task"], columns=columns, aggfunc="first"
    )[models].reindex(tasks)

    return df


def make_figure_adapter_models(
    path_to_csv: str,
    save_path: str,
    models: list,
    metric: str = "AUROC",
    rlim: str = [0.7, 1],
):
    df = (
        pd.read_csv(path_to_csv)
        .replace(
            {
                **model_names,
                **task_names,
            }
        )
        .rename(
            columns={
                "auroc": "AUROC",
                "auprc": "AUPRC",
                "auprc_c": "AUPRC_C",
                "ece": "ECE",
            }
        )
        .query("model==@models and task==@tasks")
    )

    # df = df.sort_values("task")
    # spoke_labels = df.task.unique().tolist()
    spoke_labels = tasks
    theta = radar_factory(len(tasks), frame="polygon")

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 5), subplot_kw=dict(projection="radar")
    )

    case_data = {
        # force task order to the same as labels
        x: df.query("model==@x").set_index("task").reindex(tasks)[metric].tolist()
        for x in models
    }

    for model, data in case_data.items():
        ax.plot(
            theta,
            data,
            marker="o",
            linestyle="-",
            markersize=4,
            c=color_palette[model],
            linewidth=1,
            label=model_label[model],
        )

    ax.set_varlabels(spoke_labels, **{"color": "black"})
    ax.set_title(metric)
    ax.set_rlim(rlim)

    ax.legend(ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, -0.45))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def make_few_shots_figure(
    path_to_csv_main: str,
    path_to_csv_fs: str,
    save_path: str,
    models: list,
    task: str = None,
    metric: str = "AUROC",
    figsize: tuple = (7, 4),
):
    df_adapters = pd.read_csv(path_to_csv_main)
    df_adapters_fs = pd.read_csv(path_to_csv_fs)
    df_adapters = df_adapters.replace({**model_names, **task_names}).query(
        "model==@models and task==@tasks"
    )
    df_adapters_fs = df_adapters_fs.replace({**model_names, **task_names}).query(
        "model==@models and task==@tasks"
    )
    df_adapters_fs = (
        df_adapters_fs.groupby(["model", "task", "n_shots"])[
            ["auroc", "auprc", "auprc_c", "ece"]
        ]
        .mean()
        .reset_index()
    )
    df_adapters_fs = df_adapters_fs.sort_values(["model", "n_shots"])
    df_adapters_fs["n_shots"] = df_adapters_fs["n_shots"].astype(str)
    df_adapters = df_adapters.rename(
        columns={"auroc": "AUROC", "auprc": "AUPRC", "auprc_c": "AUPRC_C", "ece": "ECE"}
    )
    df_adapters_fs = df_adapters_fs.rename(
        columns={"auroc": "AUROC", "auprc": "AUPRC", "auprc_c": "AUPRC_C", "ece": "ECE"}
    )

    if task is not None:
        df_adapters = df_adapters.query("task==@task")
        df_adapters_fs = df_adapters_fs.query("task==@task")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

    sns.lineplot(
        data=pd.concat(
            (df_adapters.assign(n_shots=x) for x in df_adapters_fs.n_shots.unique())
        ),
        x="n_shots",
        y=metric,
        hue="model",
        markers=False,
        linestyle="dashed",
        ax=ax,
        palette=color_palette,
        errorbar=None,
        err_kws={"alpha": 0.1},
        # legend=False,
    )

    sns.lineplot(
        data=df_adapters_fs,
        x="n_shots",
        y=metric,
        hue="model",
        units="task",
        markers=None,
        ax=ax,
        palette=color_palette,
        estimator=None,
        alpha=0.15,
    )

    sns.lineplot(
        data=df_adapters_fs,
        x="n_shots",
        y=metric,
        hue="model",
        style="model",
        dashes=False,
        markers=marker_style,
        markersize=7,
        ax=ax,
        palette=color_palette,
        errorbar=None,
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(metric)
    ax.set_xlabel("No. of Task-Specific Training Samples*")

    if task is not None:
        ax.set_title(task)

    ax.grid(axis="y", zorder=0, color="lightgrey")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def make_subsample_figure(
    path_to_csv_main: str,
    path_to_csv_ss: str,
    save_path: str,
    task: str = None,
    metric: str = "AUROC",
    figsize: tuple = (6, 4),
    plot_deltas: bool = False,
):
    df_adapters = pd.read_csv(path_to_csv_main)
    df_adapters_ss = pd.read_csv(path_to_csv_ss)
    df_adapters = df_adapters.replace({**model_names, **task_names}).query(
        "model=='CLMBR' and task==@tasks"
    )

    df_adapters_ss = df_adapters_ss.replace({**model_names, **task_names}).query(
        "model==['CLMBR_SK', 'CLMBR_DAPT'] and task==@tasks"
    )

    df_adapters_ss = df_adapters_ss.sort_values(["model", "perc_samples"])
    df_adapters_ss["perc_samples"] = df_adapters_ss["perc_samples"].astype(str)

    data = pd.concat(
        (
            df_adapters_ss,
            pd.concat(
                (
                    df_adapters.assign(perc_samples=x, iteration=0)
                    for x in df_adapters_ss.perc_samples
                )
            ),
        )
    )

    data = data.rename(
        columns={"auroc": "AUROC", "auprc": "AUPRC", "auprc_c": "AUPRC_C", "ece": "ECE"}
    ).drop_duplicates()
    data = data.assign(
        perc_samples=data["perc_samples"].replace(
            {
                "0.001": "0.1%\n(1.6K)\n(100K)",
                "0.01": "1%\n(16K)\n(1M)",
                "0.05": "5%\n(78K)\n(5M)",
                "0.1": "10%\n(156K)\n(10M)",
                "0.2": "20%\n(313K)\n(20M)",
                "0.4": "40%\n(626K)\n(40M)",
                "0.8": "80%\n(1.3M)\n(80M)",
            }
        )
    )

    if plot_deltas:
        models = data.model.unique()
        tasks = data.task.unique()
        base = data.query("model=='CLMBR'")

        new_data = pd.DataFrame()

        for model in models:
            for t in tasks:
                new_data = pd.concat(
                    (
                        new_data,
                        data.query("model==@model and task==@t")
                        .reset_index()
                        .assign(
                            **{
                                metric: (
                                    data.query(
                                        "model==@model and task==@t"
                                    ).reset_index()[metric]
                                    - base.query("task==@t").reset_index()[metric]
                                )
                            }
                        ),
                    )
                )

        data = new_data

    if task is not None:
        data = data.query("task==@task")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    sns.lineplot(
        data=data,
        x="perc_samples",
        y=metric,
        hue="model",
        units="task",
        markers=None,
        palette=color_palette,
        estimator=None,
        alpha=0.15,
        ax=ax,
    )

    sns.lineplot(
        data=data,
        x="perc_samples",
        y=metric,
        hue="model",
        style="model",
        dashes=False,
        markers=marker_style,
        markersize=7,
        ax=ax,
        palette=color_palette,
        errorbar=None,
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(metric)

    if plot_deltas:
        ax.set_ylabel(f"Change in {metric}")

    ax.set_xlabel(
        "Pretraining Subsample Size\n% Patients\n(No. Patients)\n(No. Coded Events)"
    )

    if task is not None:
        ax.set_title(task)

    ax.grid(axis="y", zorder=0, color="lightgrey")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
            Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
            Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def scatter(self, *args, **kwargs):
            super().scatter(*args, **kwargs)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5), num_vars, radius=0.5, edgecolor=None, zorder=0
                )
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped"""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )

                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results")
    parser.add_argument("--benchmark_tables", action="store_true")
    parser.add_argument("--benchmark_figures", action="store_true")
    parser.add_argument("--few_shots_figures", action="store_true")
    parser.add_argument("--subsample_figures", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    benchmark_tables = args.benchmark_tables or args.all
    benchmark_figures = args.benchmark_figures or args.all
    few_shots_figures = args.few_shots_figures or args.all
    subsample_figures = args.subsample_figures or args.all
    path_results = os.path.join(path_root, "results/raw/")

    if benchmark_tables:
        path_output_dir = os.path.join(path_root, "results/tables/benchmark")
        os.makedirs(path_output_dir, exist_ok=True)

        # main table
        path_to_csv = os.path.join(path_results, "adapter_models/results.csv")
        models = ["GBM", "CLMBR", "CLMBR_DAPT"]

        for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
            df = make_table_adapter_models(
                path_to_csv=path_to_csv,
                models=models,
                metric=metric,
                include_model_type=False,
            )

            df.to_csv(os.path.join(path_output_dir, f"main_{metric}.csv"))

        # supplementary table [replace GBM with CLMBR_SK]
        path_to_csv = os.path.join(path_results, "adapter_models/results.csv")
        models = ["CLMBR_SK", "CLMBR", "CLMBR_DAPT"]

        for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
            df = make_table_adapter_models(
                path_to_csv=path_to_csv,
                models=models,
                metric=metric,
                include_model_type=False,
            )

            df.to_csv(os.path.join(path_output_dir, f"supp_clmbr_sk_{metric}.csv"))

        # supplementary table [compare fine-tuning vs linear-probing]
        path_to_csv = {
            "[LP]": os.path.join(path_results, "adapter_models/results.csv"),
            "[FT]": os.path.join(path_results, "clmbr_finetuned/results.csv"),
        }

        models = ["CLMBR_SK", "CLMBR", "CLMBR_DAPT"]

        for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
            df = make_table_adapter_models(
                path_to_csv=path_to_csv,
                models=models,
                metric=metric,
                include_model_type=True,
            )

            df.to_csv(os.path.join(path_output_dir, f"supp_ft_{metric}.csv"))

    if benchmark_figures:
        # run fun for each metric
        # add supplementary (CLMBR_SK instead of GBM)
        # make_figure_adapter_models(
        #     path_to_csv: str,
        #     save_path: str,
        #     models: list,
        #     metric: str = "AUROC",
        #     ylim: str = [0.7, 1],
        # ):
        path_output_dir = os.path.join(path_root, "results/figures/benchmark")
        os.makedirs(path_output_dir, exist_ok=True)

        path_to_csv = os.path.join(path_results, "adapter_models/results.csv")
        figure_sets = {
            "main": ["GBM", "CLMBR", "CLMBR_DAPT"],
            "supp_clmbr_sk": ["CLMBR_SK", "CLMBR", "CLMBR_DAPT"],
        }

        rlims = {
            "AUROC": [0.7, 1],
            "AUPRC": [0, 0.7],
            "AUPRC_C": [0.7, 1],
            "ECE": [0, 0.03],
        }

        for figname, models in figure_sets.items():
            for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
                df = make_figure_adapter_models(
                    path_to_csv=path_to_csv,
                    save_path=os.path.join(path_output_dir, f"{figname}_{metric}.png"),
                    models=models,
                    metric=metric,
                    rlim=rlims[metric],
                )

    if few_shots_figures:
        path_output_dir = os.path.join(path_root, "results/figures/few_shots")
        os.makedirs(path_output_dir, exist_ok=True)

        path_to_csv_main = os.path.join(path_results, "adapter_models/results.csv")
        path_to_csv_fs = os.path.join(
            path_results, "adapter_models_few_shots/results.csv"
        )

        # average across tasks
        for figname, models in (
            ("main", ["GBM", "CLMBR", "CLMBR_DAPT"]),
            ("supp_clmbr_sk", ["CLMBR_SK", "CLMBR", "CLMBR_DAPT"]),
        ):
            for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
                make_few_shots_figure(
                    path_to_csv_main=path_to_csv_main,
                    path_to_csv_fs=path_to_csv_fs,
                    save_path=os.path.join(
                        path_output_dir, f"{figname}_avg_{metric}.png"
                    ),
                    models=models,
                    metric=metric,
                )

                # for each task
                for task in tasks:
                    make_few_shots_figure(
                        path_to_csv_main=path_to_csv_main,
                        path_to_csv_fs=path_to_csv_fs,
                        save_path=os.path.join(
                            path_output_dir, f"{figname}_{task}_{metric}.png"
                        ),
                        models=models,
                        task=task,
                        metric=metric,
                    )

    if subsample_figures:
        path_output_dir = os.path.join(path_root, "results/figures/subsample")
        os.makedirs(path_output_dir, exist_ok=True)

        path_to_csv_main = os.path.join(path_results, "adapter_models/results.csv")
        path_to_csv_ss = os.path.join(
            path_results, "adapter_models_subsample/results.csv"
        )

        for metric in ["AUROC", "AUPRC", "AUPRC_C", "ECE"]:
            make_subsample_figure(
                path_to_csv_main=path_to_csv_main,
                path_to_csv_ss=path_to_csv_ss,
                save_path=os.path.join(path_output_dir, f"main_avg_{metric}.png"),
                metric=metric,
                plot_deltas=True,
            )

            # for each task
            for task in tasks:
                make_subsample_figure(
                    path_to_csv_main=path_to_csv_main,
                    path_to_csv_ss=path_to_csv_ss,
                    save_path=os.path.join(
                        path_output_dir, f"main_{task}_{metric}.png"
                    ),
                    metric=metric,
                    task=task,
                    plot_deltas=True,
                )
