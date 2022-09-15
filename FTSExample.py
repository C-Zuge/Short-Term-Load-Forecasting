import warnings
warnings.filterwarnings('ignore')

import matplotlib.pylab as plt

# -------------------------------------------------------------

from pyFTS.data import Enrollments

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])

df = Enrollments.get_dataframe()
plt.plot(df['Year'],df['Enrollments'])

data = df['Enrollments'].values
# -------------------------------------------------------------

from pyFTS.partitioners import Grid

fs = Grid.GridPartitioner(data=data,npart=10)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

fs.plot(ax)
# -------------------------------------------------------------

from pyFTS.models import chen

model = chen.ConventionalFTS(partitioner=fs)
model.fit(data)
print(model)
# -------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

forecasts = model.predict(data)
forecasts.insert(0,None)

orig, = plt.plot(data, label="Original data")
pred, = plt.plot(forecasts, label="Forecasts")

plt.legend(handles=[orig, pred])

plt.show()