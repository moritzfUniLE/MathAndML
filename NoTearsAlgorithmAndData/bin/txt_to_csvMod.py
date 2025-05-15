import csv

def txt_to_csv(txt_filename, csv_filename):
    with open(txt_filename, 'r') as txt_file:
        lines = txt_file.readlines()

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #writer.writerow(['Wert1', 'Wert2'])  # Kopfzeile, optional

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    num1 = float(parts[0])
                    num2 = float(parts[1])
                    writer.writerow([num1, num2])
                except ValueError:
                    print(f"Überspringe ungültige Zeile: {line.strip()}")

