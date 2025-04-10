import pandas as pd

data = pd.read_csv('RRAM_CSV.csv')

print(data.to_string())
i = 0
area = [0]*4
energy = [0]*4
ips = [0]*4

for i in range(4):
    area[i] = data.loc[i, 'Area']
    energy[i] = data.loc[i, 'Energy']
    ips[i] = data.loc[i, 'Throughput']


print(area)
print(energy)
print(ips)