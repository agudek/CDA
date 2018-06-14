from random import random
from itertools import groupby

infected_host = "147.32.84.165"

k=[50, 100, 250, 500, 1000, 5000]
tags = []

with open("capture20110816.pcap.netflow.labeled") as file:
	file.readline() # Skip first line
	for line in file:
		values = line.split()
		# date = values[0]
		# time = values[1]
		# duration = values[2]
		# protocol = values[3]
		source_ip_port = values[4].split(":")
		source_ip = source_ip_port[0]
		# if(len(source_ip_port)>1):
		# 	source_port = source_ip_port[1]
		dest_ip_port = values[6].split(":")
		dest_ip = source_ip_port[0]
		# if(len(dest_ip_port)>1):
		# 	dest_port = source_ip_port[1]
		# flags = values[7]
		# tos = values[8]
		# packet_size = values[9]
		# flows = values[10]
		# lbl = values[11]
		# label = values[12]

		other_ip = dest_ip
		if dest_ip==infected_host:
			other_ip = source_ip

		tags.append((other_ip, random()))



real_distribution = []

for i, g in groupby(sorted(tags, key=lambda x: x[0]), key=lambda x: x[0]):
	# print(i)
	# print(list(g))
	real_distribution.append((i, len(list(g))))

real_distribution.sort(key=lambda tup: tup[1])
print("REAL DISTRIBUTION")
for (ip,count) in real_distribution[-10:]:
	print("%s | %f"%(ip, count/len(tags)))

tags.sort(key=lambda tup: tup[1])

for curr_k in k:
	distribution = []

	selected_tags = tags[:curr_k]

	for i, g in groupby(sorted(selected_tags, key=lambda x: x[0]), key=lambda x: x[0]):
		distribution.append((i, len(list(g))))

	distribution.sort(key=lambda tup: tup[1])
	print("ESTIMATED DISTRIBUTION (k=%d)"%(curr_k))
	for (ip,count) in distribution[-10:]:
		print("%s | %f"%(ip, count/curr_k))

