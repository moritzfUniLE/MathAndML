import csv
import os

def cleanUpCsv(csv_filename_input, csv_filename_output):
    with open(csv_filename_input, mode='r', encoding='utf-8') as csv_file_input, \
         open(csv_filename_output, mode='w', encoding='utf-8', newline='') as csv_file_output:
            reader = csv.reader(csv_file_input, delimiter=';')
            writer = csv.writer(csv_file_output)
            next(reader) #Skips head with protein names
            for row in reader:
                cleaned_row = [val.replace(',','.') for val in row]
                writer.writerow(cleaned_row)
if __name__ == '__main__':

    directory = '../../DataCSV/cleaned'
    dir_output = f"{directory}/conglomeratedData.csv"
    with open(dir_output, mode='w', newline='', encoding='utf-8') as csv_file_output:
        writer = csv.writer(csv_file_output)


        for file in os.listdir(directory):
            if file != "conglomeratedData.csv":
                path = f"{directory}/{file}"
                with open(path, mode='r', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file, delimiter=',')
                    for row in reader:
                        writer.writerow(row)

    # print(os.listdir('../../DataCSV/original'))