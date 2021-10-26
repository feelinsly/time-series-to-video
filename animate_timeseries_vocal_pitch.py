import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

FRAME_PERIOD = "250ms"

def ts_resample(
    s: pd.Series,
    fill_na: str,
) -> pd.Series:
    # remove duplicate timestemps 
    s = s.groupby(level=0).mean()
    # remove null
    srs = s[~s.isnull()].copy()
    srs_pad = srs.resample(FRAME_PERIOD).asfreq()
    if fill_na != "None":
        try:
            value = float(fill_na)
        except ValueError:
            value = None
            method = fill_na
        else:
            method = None
        srs_pad.fillna(value=value, method=method, inplace=True)
    return srs_pad
        
        

parser = argparse.ArgumentParser(description="timeseries to video")
parser.add_argument(
    "--csv",
    dest="csv",
    type=str,
    default="sample.csv"
)
parser.add_argument(
    "--val_name",
    dest="val_name",
    type=str,
    default="EDA"
)
parser.add_argument(
    "--val_unit",
    dest="val_unit",
    type=str,
    default="μs"
)
parser.add_argument(
    "--out",
    dest="out",
    type=str,
    default="video.mp4"
)
parser.add_argument(
    "--time_col_name",
    dest="time_col_name",
    type=str,
    default="t",
    help="the (start) time of each value",
)
parser.add_argument(
    "--end_time_col_name",
    dest="end_time_col_name",
    type=str,
    default=None,
    help="the end time of each value, by default no such column is used",
)
parser.add_argument(
    "--val_col_name",
    dest="val_col_name",
    type=str,
    default="v"
)
parser.add_argument(
    "--color_past",
    dest="color_past",
    type=str,
    default="orange",
    help="color of timeseries in the past.",
)
parser.add_argument(
    "--chart",
    dest="chart",
    type=str,
    default="line",
    help="chart type: line, area, or scatter.",
)
parser.add_argument(
    "--fill_na",
    dest="fill_na",
    type=str,
    default="pad",
    help=(
        "connect na values, with pandas fillna method,"
        " or a float number to fill,"
        " or None to not fill."
    )
)
parser.add_argument(
    "--start_frame_id",
    dest="start_frame_id",
    type=int,
    default=0,
    help=(
        "DEBUG: save plot from this frame."
    )
)
args = parser.parse_args()

dts = []
values = []

print("Reading CSV file ...")
df = pd.read_csv(args.csv)
if args.end_time_col_name:
    df["_index"] = df.index
    start_df = df[["_index", args.time_col_name, args.val_col_name]].copy()
    end_df = df[["_index", args.end_time_col_name, args.val_col_name]].copy()
    end_df.rename(columns={args.end_time_col_name: args.time_col_name}, inplace=True)
    df = pd.concat([start_df, end_df], axis=0)
    df.index = pd.DatetimeIndex(df[args.time_col_name])
    srs_pad = df.groupby("_index")[args.val_col_name].apply(ts_resample, fill_na="pad")
    srs_pad.reset_index(level="_index", drop=True, inplace=True)
else:
    df.index = pd.DatetimeIndex(df[args.time_col_name])
    srs_pad = df[args.val_col_name]

srs_pad = ts_resample(srs_pad, fill_na=args.fill_na)
print(srs_pad)
dts = srs_pad.index.to_pydatetime()
values = srs_pad.values
timepoints = [(t - dts[0]).total_seconds() for t in dts]


num_frames = len(timepoints)
frame_ids = range(num_frames)

print(f"Analyzing {args.val_name} ranges ...")

eda_upper_bounds = []
eda_lower_bounds = []
max_edas = []
min_edas = []
avg_edas = []
for frame_id in frame_ids:
    min_frame_id = max(0, frame_id - 600)
    max_frame_id = min(frame_id + 600, num_frames)

    max_eda = np.nanmax(values[min_frame_id:max_frame_id])
    min_eda = np.nanmin(values[min_frame_id:max_frame_id])
    avg_eda = np.nansum(values[min_frame_id:max_frame_id]) * 1.0 / (max_frame_id - min_frame_id)

    eda_upper_bounds.append(max((max_eda * 2.5 + 0.5) * 0.5, (avg_eda * 4 + 0.5) * 0.5))
    eda_lower_bounds.append((min_eda - abs(eda_upper_bounds[-1] - max_eda)))
    max_edas.append(max_eda)
    min_edas.append(min_eda)
    avg_edas.append(avg_eda)
    
eda_smooth_upper_bounds = []
eda_smooth_lower_bounds = []

for frame_id in frame_ids:
    min_frame_id = max(0, frame_id - 2)
    max_frame_id = min(frame_id + 2, num_frames)

    smooth_upper_eda = sum(eda_upper_bounds[min_frame_id:max_frame_id]) * 1.0 / (max_frame_id - min_frame_id)
    smooth_lower_eda = sum(eda_lower_bounds[min_frame_id:max_frame_id]) * 1.0 / (max_frame_id - min_frame_id)
    eda_smooth_upper_bounds.append(smooth_upper_eda)
    eda_smooth_lower_bounds.append(smooth_lower_eda)

pd.DataFrame(
    {
        "dts": dts,
        "values": values,
        "eda_smooth_lower_bounds": eda_smooth_lower_bounds,
        "eda_smooth_upper_bounds": eda_smooth_upper_bounds,
        "eda_upper_bounds": eda_upper_bounds,
        "eda_lower_bounds": eda_lower_bounds,
        "max_edas": max_edas,
        "min_edas": min_edas,
        "avg_edas": avg_edas,
    }
).to_csv(f"{args.out}_frame_data.csv")

plt.margins(0)
images_folder = f"{args.out}_images"
os.system(f"rm -rf {images_folder} && mkdir {images_folder}")

ylabel = f"{args.val_name} ({args.val_unit})" if args.val_unit else args.val_name

for frame_id in frame_ids:
    print(f"Generating frame {frame_id + 1} / {num_frames}")
    if frame_id < args.start_frame_id:
        print("skipped")
        continue
    frame_time = frame_id * 0.25

    min_frame_id = max(0, frame_id - 600)
    max_frame_id = min(frame_id + 600, num_frames)

    min_time = frame_time - 150
    max_time = frame_time + 150

    eda_lower_bound = eda_smooth_lower_bounds[frame_id]
    eda_upper_bound = eda_smooth_upper_bounds[frame_id]
    
    if args.chart == "area":
        plt.fill_between(
            timepoints[min_frame_id:frame_id], 
            values[min_frame_id:frame_id],
            color=args.color_past, 
        )
        plt.fill_between(
            timepoints[frame_id:max_frame_id], 
            values[frame_id:max_frame_id],
            color=(0.5, 0.5, 0.5), 
            alpha=0.4,
        )
    elif args.chart == "line":
        plt.plot(timepoints[min_frame_id:frame_id], values[min_frame_id:frame_id], '-',
                 linewidth = 3, color = args.color_past)
        plt.plot(timepoints[frame_id:max_frame_id], values[frame_id:max_frame_id], '-',
                 linewidth = 3, color = (0.5, 0.5, 0.5))
    elif args.chart == "scatter":
        plt.plot(timepoints[min_frame_id:frame_id], values[min_frame_id:frame_id], '*',
                 linewidth = 3, color = args.color_past)
        plt.plot(timepoints[frame_id:max_frame_id], values[frame_id:max_frame_id], '*',
                 linewidth = 3, color = (0.5, 0.5, 0.5))
    else:
        raise ValueError(f"Not a supported chart type: {args.chart}")

    plt.ylabel(ylabel, fontsize = 12, labelpad = 11)

    fig = plt.gcf()

    axes = fig.gca()
    axes.get_xaxis().set_visible(False)

    axes.set_xlim([min_time, max_time])
    axes.set_ylim([eda_lower_bound, eda_upper_bound])

    rect = patches.Rectangle(
        (frame_time, eda_lower_bound),
        max_time - frame_time,
        eda_upper_bound - eda_lower_bound,
        color=(0.9, 0.9, 0.9), 
        fill=True,
        alpha=0.5,
    )
    axes.add_patch(rect)

    fig.text(
        0.27, 
        0.13,
        f"{args.val_name} = {'%.3f' % values[frame_id]}{args.val_unit}",
        fontsize = 8,
        fontname = 'Menlo',
        horizontalalignment='center',
        verticalalignment='top',
    )

    fig.text(0.48, 0.06,
             f"{dts[frame_id].strftime('%H:%M:%S')}",
             fontsize = 12,
             fontname = 'Menlo')

    fig.set_size_inches(12, 6)

    filename = f"{images_folder}/image-" + str(frame_id + 1 - frame_ids[0]).rjust(len(str(num_frames)), "0") + ".png"
    fig.savefig(filename)

    plt.clf()

os.system(f"rm -rf {args.out}")
os.system(f"ffmpeg -r 4 -i {images_folder}/image-%0{len(str(num_frames))}d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p {args.out}")
