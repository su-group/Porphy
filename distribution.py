import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
df_PBDD = pd.read_csv('PBDD.csv')
df_porphyrin = pd.read_csv('MpPD.csv')
sns.kdeplot(df_PBDD['HOMO'], color='#6590B7', shade='True')
sns.kdeplot(df_porphyrin['HOMO'], color='#F7CF8D', shade='True')
sns.kdeplot(df_PBDD['LUMO'], color='#466F94', shade='True')
sns.kdeplot(df_porphyrin['LUMO'], color='#F1A837', shade='True')
sns.kdeplot(df_PBDD['E_gap'], color='#273E53', shade='True')
sns.kdeplot(df_porphyrin['E_gap'], color='#BB790D', shade='True')
plt.xlim(-8, 6)
plt.ylim(-0.05, )
plt.xlabel("Energy (eV)", fontsize=12, labelpad=10)
plt.ylabel("Distribution", fontsize=15, labelpad=10)
# plt.title('Energy Distribution in PBDD and MpPD', fontsize=15)
# plt.legend(labels=['PBDD', "MpPD"])
# plt.show()
plt.savefig('distribution.png', dpi=300)
