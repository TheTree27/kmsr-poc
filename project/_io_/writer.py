import csv

def write_header(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["centers","radii","sum_of_radii"])

def write_result(filename, centers, radii, sum_of_radii):
    row =  [centers, radii, sum_of_radii]
    with open(filename, mode='a', newline='') as file: # mode a to append
        writer = csv.writer(file)
        writer.writerow(row)