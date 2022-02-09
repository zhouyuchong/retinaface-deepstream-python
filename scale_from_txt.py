def get_network_scale(filename):
    f = open(filename)
    lines = f.readlines()
    for line in lines:
        if line[:12] == "input-height": 
            height = line.split(":")[1]
        if line[:11] == "input-width":
            width = line.split(":")[1]
    f.close()
    return int(height), int(width)