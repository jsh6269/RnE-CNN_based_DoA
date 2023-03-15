import json
file_name = 't3_gt.json'
angle_tag = [0,20,40,60,80,100,120,140,160,180,-1]
data = {}
id = 0
data["track3_results"] = []
print("- Enter the number of files -")

for i in range(11):
    temp = int(input("%d degrees : " %angle_tag[i]))
    for _ in range(temp):
        data["track3_results"].append({
            "angle": angle_tag[i],
            "id": id
        })
        id+=1

with open(file_name,'w') as outfile:
    json.dump(data, outfile, indent=4)
print("'"+file_name+"'", 'is successfully made')

