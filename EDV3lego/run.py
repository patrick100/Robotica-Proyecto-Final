#!/usr/bin/env python3
from ev3dev2.sensor.lego import TouchSensor, ColorSensor
from ev3dev2.sound import Sound
from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B

from time import sleep
import socket

R = 30
L = 30

s = socket.socket()          
print("Socket successfully created")
# reserve a port on your computer in our 
# case it is 12345 but it can be anything 
port = 12345

def turn(i):
        if (i == 'r'):
                tank_pair.on(left_speed=-R, right_speed=L)
        elif (i == 'l'):
                tank_pair.on(left_speed=R, right_speed=-L)
        sleep(0.8)

def go(i):
        if (i == 'f'):
                tank_pair.on(left_speed=R, right_speed=L)
        elif (i == 'i'):
                tank_pair.on(left_speed=-R, right_speed=-L)
#       sleep(t)

def stop():
        tank_pair.off()


cl = ColorSensor()
sound = Sound()

tank_pair = MoveTank(OUTPUT_A, OUTPUT_B)

go('f')

print(str(cl.color)+" "+cl.color_name )
c = cl.color
while(c!=5):

        print(str(cl.color)+" "+cl.color_name )
        c = cl.color
        if(c==5):
               s.connect(("192.168.43.167", port))

               print (s.recv(1024))

               stop()
               sleep(2)
               go('f')
               turn('l')
               break
               sound.speak(cl.color_name)

go('f')
sleep(4)
stop()
s.close()
