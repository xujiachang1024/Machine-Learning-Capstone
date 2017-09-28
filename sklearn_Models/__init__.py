# Import dependencies
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning


# Set the seed (reproducibility)
random.seed(0)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


# Data import and cleaning
df1 = pd.read_csv("./speedbumps_1.csv")  # read data from the .csv file
df2 = pd.read_csv("./speedbumps_2.csv")  # read data from the .csv file
df3 = pd.read_csv("./speedbumps_3.csv")  # read data from the .csv file
df4 = pd.read_csv("./speedbumps_4.csv")  # read data from the .csv file
df = pd.read_csv("./speedbumps_5.csv")  # read data from the .csv file
df1 = df1.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df2 = df2.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df3 = df3.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df4 = df4.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df = df.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df = df.append(df1)
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)
keywords = ['yes', 'no']
mapping = [1, 0]
df = df.replace(keywords, mapping)