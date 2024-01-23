import pandas as pd
import wandb


def fetch_full_data_from_wandb(run_path, key1, key2):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.scan_history(keys=[key1, key2], page_size=10000)

    key1_list = []
    key2_list = []

    for row in history:
        key1_list.append(row[key1])
        key2_list.append(row[key2])

    dictionary = {key1: key1_list, key2: key2_list}
    df = pd.DataFrame.from_dict(dictionary)
    return df


if __name__ == "__main__":
    df = fetch_full_data_from_wandb("tlips/cloth-keypoints-paper/22dtzjq1", "epoch", "test/meanAP/d=2.0")

    print(df)
    df.to_csv("keypoint-detection.csv")
