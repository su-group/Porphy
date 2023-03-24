import numpy as np
import pandas as pd
import tmap as tm
from faerun import Faerun
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from tqdm import tqdm

structure_labels = [
    (0, 'PBDD'),
    (1, 'MpPD'),
]

# colors = ['blue', 'purple', 'green']
# cmap = matplotlib.colors.ListedColormap(colors)

df1 = pd.read_csv("/home/dell/PycharmProjects/zx-all-model/PBDD.csv")
df1['ori'] = 0
df2 = pd.read_csv('/home/dell/PycharmProjects/zx-all-model/MpPD.csv')
df2['ori'] = 1
df = pd.concat([df1, df2])
# df = df[:5000].copy()
list_main_smi = df['smiles'].tolist()
list_main_num = df['ori'].tolist()
list_gap = df['E_gap'].tolist()

# c_list = df['EGP'].tolist()
# compute reaction fingerprint
model, tokenizer = get_default_model_and_tokenizer('pfas_config')
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
pfas_fp = generate_fingerprints(list_main_smi, rxnfp_generator, batch_size=16)
np.savez_compressed('pfas', fps=pfas_fp)
pfas_fp = np.load('pfas.npz')['fps']

labels = []
for index, i in df.iterrows():
    label = (
            i['smiles'] +
            "__<h1>" + str(i['E_gap']) + "</h1>"
    )
    labels.append(label)

dims = 256
enc = tm.Minhash(dims)

lf = tm.LSHForest(dims, 128)
cfg = tm.LayoutConfiguration()
cfg.k = 150
cfg.kc = 150
cfg.sl_scaling_min = 1.0
cfg.sl_scaling_max = 1.0
cfg.sl_repeats = 1
cfg.sl_extra_scaling_steps = 2
cfg.placer = tm.Placer.Barycenter
cfg.merger = tm.Merger.LocalBiconnected
cfg.merger_factor = 2.0
cfg.merger_adjustment = 0
cfg.fme_iterations = 100
cfg.sl_scaling_type = tm.ScalingType.RelativeToDesiredLength
cfg.node_size = 1 / 37
cfg.mmm_repeats = 1

# fingerprints = [tm.VectorUint(enc.encode(s)) for s in list_main_smi]
fingerprints = [enc.from_weight_array(fp.tolist(), method="I2CWS") for fp in tqdm(pfas_fp)]
lf.batch_add(fingerprints)
lf.index()
x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)
name = 'PBDD_and_porphyrin'
faerun = Faerun(coords=False, view='front', )
faerun.add_scatter(
    name,
    {"x": x,
     "y": y,
     "c": list_main_num,
     "labels": labels},
    point_scale=2.5,
    max_point_size=20,
    colormap="rainbow",
    has_legend=True,
    legend_labels=structure_labels,
    categorical=False,
    # series_title="dGsolv",
    shader='smoothCircle'
)
faerun.add_tree(name + "_tree", {"from": s, "to": t}, point_helper=name)

faerun.plot(name, template="smiles")
