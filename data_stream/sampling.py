from random import random
from itertools import groupby
from linkedlist import llist   
import time
from sys import getsizeof

infected_host = "147.32.84.165"

k = 5000

top = llist()

with open("capture20110816.pcap.netflow.labeled") as file:
	file.readline() # Skip first line
	start_time = time.time()
	for line in file:
		values = line.split()
		source_ip_port = values[4].split(":")
		source_ip = source_ip_port[0]
		dest_ip_port = values[6].split(":")
		dest_ip = dest_ip_port[0]

		if source_ip==infected_host:
			other_ip = dest_ip
		elif dest_ip==infected_host:
			other_ip = source_ip
		else:
			continue

		val = random()

		new_node = top.root
		while new_node != None:
			if val<new_node.value:
				top.insert(new_node.prev, new_node, other_ip, val)
				break
			new_node = new_node.next

		if top.length > k:
			top.pop()

		if top.root == None:
			top.insert(None, None, other_ip, val)


distribution = []
for i, g in groupby(sorted(top.aslist(), key=lambda x: x[0]), key=lambda x: x[0]):
	distribution.append((i, len(list(g))))

distribution.sort(key=lambda tup: tup[1], reverse=True)

print("--- %d bytes ---"%(getsizeof(top)+getsizeof(distribution)))
print("--- %s seconds ---\n" % (time.time() - start_time))

print("ESTIMATED DISTRIBUTION (k=%d)"%(k))
for (ip,count) in distribution[:10]:
	print("%s,%f"%(ip, count/k))
