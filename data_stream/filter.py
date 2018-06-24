import pandas as pd

cols = ["date","time", "duration", "protocol", "source_ip", "source_port", "destination_ip", "destination_port", "flags", "tos", "packets", "bytes", "flows", "label"]

data = {}

'''Open file and parse'''
with open("capture20110818.pcap.netflow.labeled") as file:
	file.readline() # Skip first line
	i = 0
	for line in file:
		values = line.split()

		source_ip_port = values[4].split(":")
		source_ip = source_ip_port[0]
		source_port = -1
		if(len(source_ip_port)>1):
			source_port = source_ip_port[1]
		
		dest_ip_port = values[6].split(":")
		dest_ip = dest_ip_port[0]
		dest_port = -1
		if(len(dest_ip_port)>1):
			dest_port = dest_ip_port[1]
		
		values[4] = source_ip
		values.insert(5,source_port)

		values.pop(6)

		values[6] = dest_ip
		values.insert(7,dest_port)

		
		label = values[13]

		# Add to dict if not a background flow
		if label != 'Background':
			data[i] = dict(zip(cols, values))
			i+=1

# Convert to dataframe and write to csv file
df = pd.DataFrame.from_dict(data, "index")
df.to_csv("scenario_10_filtered.csv")