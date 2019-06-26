# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:18:31 2019

@author: robert
"""

from __future__ import print_function
import psutil
import time

def write_new_data():
    f = open('devices_information.csv', 'a', newline='')
    ticks = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    print(ticks)


    cpu_usage_rate = psutil.cpu_percent()
    print("CPU 使用率:",cpu_usage_rate)

    #print(cpu_usage_rate)
    print("------------------")
    #print(psutil.virtual_memory())  # physical memory usage
    print("Memory 使用率:")
    momery_usage_rate = psutil.virtual_memory()[2]
    print('memory % used:', momery_usage_rate)


    writer = f.write(str(cpu_usage_rate)+","+str(momery_usage_rate)+","+str(ticks)+'\n')
    time.sleep(5)




if __name__ == "__main__":
    while True:
        write_new_data()