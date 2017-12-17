import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def get_training_stats(filename):
    angles = defaultdict(int)
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for line in reader:
            angle = float(line[3][:6])
            angles[angle] += 1
    return angles



data = get_training_stats('../recovery_driving/driving_log.csv')


lists = sorted(data.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.title('Distribution of steering angles while training for recovery driving')
plt.savefig('recovery_angles.png')
