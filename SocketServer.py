# The socket server to communicate with the Android Studio Application and read in the images taken by the user

import socket
import struct
import preprocessing as process

# define server values
listensocket = socket.socket()
Port = 8000
maxConnections = 999
IP = socket.gethostname()  # get hostname of current machine

listensocket.bind(('', Port))

# open server
listensocket.listen(maxConnections)
print("Server started at " + IP + " on port " + str(Port))

# accepts incoming connection
(clientsocket, address) = listensocket.accept()
print("New connection made!")

buf = ''
while len(buf) < 4:
    buf += clientsocket.recv(4 - len(buf))
size = struct.unpack('!i', buf)
print("receiving %s bytes", size)
with open('test.jpg', 'wb') as img:
    while True:
        data = clientsocket.recv(1024) # receive image
        if not data:
            break
        img.write(data)

print("received image!")
process.imgpreprocessing(data)

clientsocket.close()
