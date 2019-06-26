import csv
import matplotlib.pyplot as plt
from datetime import datetime 

def drowing():
    cup_usage = []
    momery_usage = []
    time = []
    with open('devices_information.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            cup_usage.append(row['CUP_usage_rate'])
            momery_usage.append(row['Momery_usage_rate'])

        plt.plot(cup_usage, color='red', label='cup_usage')
        plt.xticks(fontsize=2)
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(momery_usage, color='blue', label='momery_usage')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    drowing()