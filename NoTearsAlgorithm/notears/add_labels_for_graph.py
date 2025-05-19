import csv
from os import path

def addLabels(input_CSV):
    base_name = path.splitext(input_CSV)[0]
    output_CSV = f"{base_name}_labeled.csv"
    with open(input_CSV, mode='r', newline='', encoding='utf-8') as csvfilein, \
         open(output_CSV, mode='w', newline='', encoding='utf-8') as csvfileout:
            reader = csv.reader(csvfilein, delimiter=',')
            writer = csv.writer(csvfileout, delimiter=',')
            head = ["protein","Raf","Mek","Plcg","PIP2","PIP3","Erk","Akt","PKA","PKC","P38","Jnk"]
            writer.writerow(head)
            for i,row in enumerate(reader):
                protein = head[i+1]
                row_new = [protein]
                row_new.extend(row)
                writer.writerow(row_new)

    return output_CSV

if __name__ == "__main__":
    addLabels("../W_est_nonlinear_MainData.csv")