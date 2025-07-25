import numpy as np
import re

def read(filename):
    points = []

    # lots of basic IO that was very annoying
    with open(filename, 'r') as data:
        for line in data:
            line = line.strip()
            # skip empty lines
            if not line:
                continue

            # split on either comma or space, depending on what is in the file. might break if floats are
            # seperated by commas (fortunately they aren't in the /data_sets/ folder)
            coordinates = re.split(r'[,\s]+', line)

            current_point = []
            for coordinate in coordinates:
                try:
                    current_point.append(float(coordinate))
                except ValueError:
                    continue  # skip non-numeric elements (so strings), necessary for iris classifications
            # at the end of the line if it's not empty add the point to the data
            if current_point:
                points.append(current_point)

    if points:
        return np.array(points, dtype=float)
    else:
        raise Exception("Input file did not contain numeric points")