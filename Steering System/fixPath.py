path = '/Users/mohammedamarnah/Desktop/SDCProject/data/driving_log.csv'

file = open(path, 'r')

data = file.readlines()

i = 1
while (i <= len(data)):
	print(i)
	i++