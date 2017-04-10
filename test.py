f = open('model_param.txt', 'r')

layers = []

for line in f:
    if len(line) > 1 and line[0] != "#":
        raw_layer = line[:-1].lower().split(",")
        raw_layer = [x.strip() for x in raw_layer]
        raw_layer[1] = int(raw_layer[1])
        raw_layer[2] = int(raw_layer[2])
        raw_layer[3] = True if raw_layer[3] == "true" else False
        raw_layer[4] = "" if raw_layer[4] == "none" else raw_layer[4]

        layers.append(raw_layer)
        
print(layers)
