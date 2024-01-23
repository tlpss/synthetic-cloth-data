from state_estimation.keypoint_detection.results.wandb_pull import fetch_full_data_from_wandb
from state_estimation.keypoint_detection.results.wandb_runs import (
    SHORTS_REAL_RUN,
    SHORTS_SIM_FINETUNED_RUN,
    SHORTS_SIM_RUN,
    SHORTS_SIM_SIM_RUN,
    TOWELS_REAL_RUN,
    TOWELS_SIM_FINETUNED_RUN,
    TOWELS_SIM_RUN,
    TOWELS_SIM_SIM_RUN,
    TSHIRTS_REAL_RUN,
    TSHIRTS_SIM_FINETUNED_RUN,
    TSHIRTS_SIM_RUN,
    TSHIRTS_SIM_SIM_RUN,
)


def get_AP_metrics(run):
    df = fetch_full_data_from_wandb(run, "epoch", "test/meanAP/meanAP")
    mAPmean = df["test/meanAP/meanAP"][0]
    df = fetch_full_data_from_wandb(run, "epoch", "test/meanAP/d=2.0")
    mAP2 = df["test/meanAP/d=2.0"][0]
    return mAPmean, mAP2


real_tshirts_aps = get_AP_metrics(TSHIRTS_REAL_RUN)
real_towels_aps = get_AP_metrics(TOWELS_REAL_RUN)
real_shorts_aps = get_AP_metrics(SHORTS_REAL_RUN)

sim_tshirts_aps = get_AP_metrics(TSHIRTS_SIM_RUN)
sim_towels_aps = get_AP_metrics(TOWELS_SIM_RUN)
sim_shorts_aps = get_AP_metrics(SHORTS_SIM_RUN)

sim_finetuned_tshirts_aps = get_AP_metrics(TSHIRTS_SIM_FINETUNED_RUN)
sim_finetuned_towels_aps = get_AP_metrics(TOWELS_SIM_FINETUNED_RUN)
sim_finetuned_shorts_aps = get_AP_metrics(SHORTS_SIM_FINETUNED_RUN)

sim_sim_tshirts_aps = get_AP_metrics(TSHIRTS_SIM_SIM_RUN)
sim_sim_towels_aps = get_AP_metrics(TOWELS_SIM_SIM_RUN)
sim_sim_shorts_aps = get_AP_metrics(SHORTS_SIM_SIM_RUN)

real_avg_aps = (
    (real_towels_aps[0] + real_shorts_aps[0] + real_tshirts_aps[0]) / 3,
    (real_towels_aps[1] + real_shorts_aps[1] + real_tshirts_aps[1]) / 3,
)
sim_avg_aps = (
    (sim_towels_aps[0] + sim_shorts_aps[0] + sim_tshirts_aps[0]) / 3,
    (sim_towels_aps[1] + sim_shorts_aps[1] + sim_tshirts_aps[1]) / 3,
)
sim_finetuned_avg_aps = (
    (sim_finetuned_towels_aps[0] + sim_finetuned_shorts_aps[0] + sim_finetuned_tshirts_aps[0]) / 3,
    (sim_finetuned_towels_aps[1] + sim_finetuned_shorts_aps[1] + sim_finetuned_tshirts_aps[1]) / 3,
)
sim_sim_avg_aps = (
    (sim_sim_towels_aps[0] + sim_sim_shorts_aps[0] + sim_sim_tshirts_aps[0]) / 3,
    (sim_sim_towels_aps[1] + sim_sim_shorts_aps[1] + sim_sim_tshirts_aps[1]) / 3,
)
table = (
    "\toprule\n"
    "\tCloth Type & \multicolumn{3}{c}{\APAll} & \multicolumn{3}{c}{\APTwo} \\\\\n"
    "\t\cmidrule(lr){2-4}\n"
    "\t\cmidrule(lr){5-7}\n"
    "\t& real2real & sim2real & sim+real2real  & real2real& sim2real & sim+real2real  \\\\ \n"
    "\t\midrule\n"
    f"\tTshirt & {real_tshirts_aps[0]:.1f} & {sim_tshirts_aps[0]:.1f} &{sim_finetuned_tshirts_aps[0]:.1f}  &{real_tshirts_aps[1]:.1f} & {sim_tshirts_aps[1]:.1f} & {sim_finetuned_tshirts_aps[1]:.1f} \\\\ \n"
    f"\tTowel & {real_towels_aps[0]:.1f}  & {sim_towels_aps[0]:.1f} & {sim_finetuned_towels_aps[0]:.1f} & {real_towels_aps[1]:.1f} & {sim_towels_aps[1]:.1f} & {sim_finetuned_towels_aps[1]:.1f} \\\\ \n"
    f"\tShorts & {real_shorts_aps[0]:.1f} & {sim_shorts_aps[0]:.1f} & {sim_finetuned_shorts_aps[0]:.1f} & {real_shorts_aps[1]:.1f} & {sim_shorts_aps[1]:.1f} & {sim_finetuned_shorts_aps[1]:.1f} \\\\ \n"
    "\t\midrule\n"
    f"\All & {real_avg_aps[0]:.1f} & {sim_avg_aps[0]:.1f} & {sim_finetuned_avg_aps[0]:.1f} & {real_avg_aps[1]:.1f} & {sim_avg_aps[1]:.1f} & {sim_finetuned_avg_aps[1]:.1f} \\\\ \n"
)

table = (
    "begin{tabular}{lcccc} \n"
    "\t \\toprule\\ \n"
    "\tCloth Type & \multicolumn{3}{c}{\APAll} & \multicolumn{3}{c}{\APTwo} \\\\\n"
    "\t\cmidrule(lr){2-4}\n"
    "\t\cmidrule(lr){5-7}\n"
    "\t& real & sim & sim + real  & real & sim & sim + real  \\\\ \n"
    "\t\midrule\n"
    f"\tTshirt & {real_tshirts_aps[0]*100:.1f} & {sim_tshirts_aps[0]*100:.1f} &{sim_finetuned_tshirts_aps[0]*100:.1f}  &{real_tshirts_aps[1]*100:.1f} & {sim_tshirts_aps[1]*100:.1f} & {sim_finetuned_tshirts_aps[1]*100:.1f} \\\\ \n"
    f"\tTowel & {real_towels_aps[0]*100:.1f}  & {sim_towels_aps[0]*100:.1f} & {sim_finetuned_towels_aps[0]*100:.1f} & {real_towels_aps[1]*100:.1f} & {sim_towels_aps[1]*100:.1f} & {sim_finetuned_towels_aps[1]*100:.1f} \\\\ \n"
    f"\tShorts & {real_shorts_aps[0]*100:.1f} & {sim_shorts_aps[0]*100:.1f} & {sim_finetuned_shorts_aps[0]*100:.1f} & {real_shorts_aps[1]*100:.1f} & {sim_shorts_aps[1]*100:.1f} & {sim_finetuned_shorts_aps[1]*100:.1f} \\\\ \n"
    "\t\midrule\n"
    f" All & {real_avg_aps[0]*100:.1f} & {sim_avg_aps[0]*100:.1f} & {sim_finetuned_avg_aps[0]*100:.1f} & {real_avg_aps[1]*100:.1f} & {sim_avg_aps[1]*100:.1f} & {sim_finetuned_avg_aps[1]*100:.1f} \\\\ \n"
    "\t\\bottomrule\n"
    "\end{tabular}"
)

print("main table")
print(table)


sim_vs_real_table = (
    "\\begin{tabular}{lcccc} \n"
    "\\toprule \\ \n"
    "\\textbf{cloth type} & \multicolumn{2}{c}{\APAll} & \multicolumn{2}{c}{\APTwo} \\\\ \n"
    "\cmidrule(lr){2-3}\n"
    "\cmidrule(lr){4-5}\n"
    "& sim2real & sim2sim  & sim2real & sim2sim  \\\\ \n"
    "\midrule\n"
    f"Tshirt & {sim_tshirts_aps[0]*100:.1f} & {sim_sim_tshirts_aps[0]*100:.1f} & {sim_tshirts_aps[1]*100:.1f} & {sim_sim_tshirts_aps[1] *100:.1f} \\\\n"
    f"Towel & {sim_towels_aps[0]*100:.1f} & {sim_sim_towels_aps[0]*100:.1f} & {sim_towels_aps[1]*100:.1f} & {sim_sim_towels_aps[1] *100:.1f} \\\\ \n"
    f"Shorts & {sim_shorts_aps[0]*100:.1f} & {sim_sim_shorts_aps[0]*100:.1f} & {sim_shorts_aps[1]*100:.1f} & {sim_sim_shorts_aps[1] *100:.1f} \\\\ \n"
    "\midrule\n"
    f"All & {sim_avg_aps[0]*100:.1f} & {sim_sim_avg_aps[0]*100:.1f} & {sim_avg_aps[1]*100:.1f} & {sim_sim_avg_aps[1] *100:.1f} \\\\ \n"
    "\\bottomrule\n"
    "\end{tabular}"
)
print("sim vs real table")
print(sim_vs_real_table)
