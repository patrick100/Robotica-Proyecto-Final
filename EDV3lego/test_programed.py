#!/usr/bin/env python3
from ev3dev2.sensor.lego import TouchSensor, ColorSensor
from ev3dev2.sound import Sound
from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B
#from ev3dev2.sensor.lego import TouchSensor
from time import sleep
import threading
import socket

R = 30
L = 30
cc = 1
#actions = ["r", "l", "s"]
#count = 0

def turn(i):
        if (i == 'r'):
                tank_pair.on(left_speed=-R, right_speed=L)
                sleep(0.8)
        elif (i == 'l'):
                tank_pair.on(left_speed=R, right_speed=-L)
                sleep(0.9)
#        tank_pair.off()

def go(i):
        if (i == 'f'):
                tank_pair.on(left_speed=R, right_speed=L)
        elif (i == 'i'):
                tank_pair.on(left_speed=-R, right_speed=-L)

def stop():
        tank_pair.off()


def myserver():
        # next create a socket object
        s = socket.socket()
        print("Socket successfully created")

        # reserve a port on your computer in our
        # case it is 12345 but it can be anything
        port = 12345
        s.bind(("", port))
        print("socket binded to %s" %(port))
        # put the socket into listening mode
        s.listen(5)
        print("socket is listening")

        # a forever loop until we interrupt it or
        # an error occurs
        actions = ["r", "l", "s"]
        count = 0
        while True:

                # Establish connection with client. 
                c, addr = s.accept() 

                data = c.recv(1024).decode()
                if not data:
                        print("not data: " +(data))
                        break
                else:
                        #turn('l')
                        print("Recieved: "+(data))
                        data = actions[count]
                        count = count + 1
                        if (data != "s"):
                        #stop()
                        #sleep(2)
                                go('f')
                                turn(data)
                                go('f')
                                sleep(2)
                        else:
                                stop()
                        #break
                        #sound.speak(cl.color_name)
                #print 'Got connection from', addr 

                # send a thank you message to the client.  
                #c.send('Thank you for connecting') 

                # Close the connection with the client 
                c.close()


def main():

        #3 verde, 4 amarillo, 5 rojo, 6 blanco
        cl = ColorSensor()
        sound = Sound()

        tank_pair = MoveTank(OUTPUT_A, OUTPUT_B)
        go('f')
        #inicializamos el servidor como un thread

        hilo = threading.Thread(target=myserver)
        hilo.start()

        #una vez que la imagen se guarde llega la respuesta al servidor

        #enviamos la instruccion a ejecutar en el robot


        #cuando reconozca el sensor de color

        # Create a socket object
        #s = socket.socket()
        #port = 12345
        # ip phone
        #s.connect(("192.168.43.167", port))
        #s.close()
        print(str(cl.color)+" "+cl.color_name )
        while True:

                print(str(cl.color)+" "+cl.color_name )
                cc = cl.color
                if(cc==5):
                        s = socket.socket()
                        port = 12345
                        stop()
                        #sleep(2)
                        s.connect(("192.168.43.91", port))
                        sleep(5)
                        cc = 1
                        #break
                        #stop()
                        #sleep(2)
                        #s.close()
                        #print (s.recv(1024))
                        #stop()
                        #sleep(2)
                        #go('f')
                        #turn('l')
                        #break
                        #sound.speak(cl.color_name)


        hilo.join()


cl = ColorSensor()
sound = Sound()

tank_pair = MoveTank(OUTPUT_A, OUTPUT_B)
main()


#3 verde, 4 amarillo, 5 rojo, 6 blanco
#cl = ColorSensor()
#sound = Sound()

#tank_pair = MoveTank(OUTPUT_A, OUTPUT_B)
#go('f')


#print(str(cl.color)+" "+cl.color_name )
#c = cl.color
#while(c!=5):

        #print(str(cl.color)+" "+cl.color_name )
        #c = cl.color
        #if(c==5):
        #       s.connect(("192.168.0.31", port))


# receive data from the server
        #       print (s.recv(1024))

# close the connection
        #       stop()
        #       sleep(2)
        #       go('f')
        #       turn('l')
        #       break
        #       sound.speak(cl.color_name)

#go('f')
#sleep(4)
#stop()
#s.close()
