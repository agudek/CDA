import mmh3
import sys
from linkedlist import llist   
import time
from sys import getsizeof

w = 250
d = 3

def _get_hashes(val):
    h = []
    for i in range(d):
        if i < 2:
            h_i = mmh3.hash(val, 1984+i, signed=False)%w
        else:
            # Kirsch-Mitzenmacher-Optimization to minimise number of hashing operations
            h_i = ((h[0] + i*h[1]) % w)
        h.append(h_i)
    return h

def sketch(mat, val):
    for i,h in enumerate(_get_hashes(val)):
        mat[h][i] += 1
    

def get_count(mat, val):
    min_cnt = sys.maxsize
    for i,h in enumerate(_get_hashes(val)):
        if mat[h][i] < min_cnt:
            min_cnt = mat[h][i]
    return min_cnt

infected_host = "147.32.84.165"
mat = []
for i in range(w):
    mat.append([0]*d)

top = llist()

n=0
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
            n+=1
            other_ip = dest_ip
        elif dest_ip==infected_host:
            n+=1
            other_ip = source_ip
        else:
            continue

        sketch(mat, bytes(other_ip,'utf-8'))

        val = get_count(mat, bytes(other_ip,'utf-8'))

        new_node = top.root
        while new_node != None:
            if val>=new_node.value:
                top.insert(new_node.prev, new_node, other_ip, val)
                break
            new_node = new_node.next

        while new_node!= None:
            if new_node.key == other_ip:
                top.remove(new_node)
            new_node = new_node.next

        if top.length > 10:
            top.pop()

        if top.root == None:
            top.insert(None, None, other_ip, val)

print("--- %d bytes ---"%(getsizeof(top)+getsizeof(mat)))
print("--- %s seconds ---\n" % (time.time() - start_time))


new_node = top.root
print("sketch (w=%d, d=%d)"%(w,d))
while new_node != None:
    print("%s,%f"%(new_node.key, new_node.value/n))
    new_node =  new_node.next