import pandas as pd

df = pd.read_pickle(
    "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/{}_1101_1115_data_sample10w.pkl".format(
        "chengdu", "chengdu"
    )
)
df.to_parquet(
    "/home/harddisk/jxh/trajectory/JGRM/dataset/{}/data.parquet".format("chengdu")
)
