import os
import pandas as pd

def export_df(
    df: pd.DataFrame,
    output_path: str,
    overwrite: bool = False,
    header: bool = True,
):
    if not os.path.isfile(output_path) or overwrite:
        parent_path = os.path.dirname(output_path)

        if os.path.exists(parent_path):
            df.to_csv(output_path, index=False, header=header)
            print("Saved dataset: %s", output_path)
        else:
            print(f"{output_path} path does not exist")
    else:
        print("file already exists")