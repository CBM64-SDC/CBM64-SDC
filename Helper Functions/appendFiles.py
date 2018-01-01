import sys

# Use to append 2 files like csv for datasets
# python appendFiles.py path/to/input/file1 path/to/input/file2 path/to/outputFile

file1 = sys.argv[1]
file2 = sys.argv[2]
outputFile = sys.argv[3] 
files = [file1, file2]

with open(outputFile, 'w') as outfile:
	for x in files:
		with open(x) as infile:
			for line in infile:
				outfile.write(line)
