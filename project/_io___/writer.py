import csv



def write_header(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["k", "centers","radii","sum_of_radii", "data_set"])

def write_result(filename, k, centers, radii, sum_of_radii, algorithm):
    row =  [k, centers, radii, sum_of_radii, algorithm]
    with open(filename, mode='a', newline='') as file: # mode a to append
        writer = csv.writer(file)
        writer.writerow(row)