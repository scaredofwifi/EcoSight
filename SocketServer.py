# The socket server to communicate with the Android Studio Application and read in the images taken by the user

import socket
from train_test import Demo
# import preprocessing as process

# set up server listener
listensocket = socket.socket()
Port = 8000
maxConnections = 999
IP = socket.gethostname()  # get hostname of current machine

listensocket.bind(('', Port))

# start server
listensocket.listen(maxConnections)
print("Server started at " + IP + " on port " + str(Port))

d = Demo()
print("[status] classification model started")
# accepts incoming connection
(clientsocket, address) = listensocket.accept()
print("New connection made!")

# open file
f = open('incoming.jpg', 'wb')  # TODO : change the filepath of this incoming image
data_in = 1

# receive image
while data_in:
    data_in = clientsocket.recv(1024)  # get incoming data
    if not data_in:
        break
    f.write(data_in)  # writes data to file



# process image
# process.imgpreprocessing(data)

# TODO : send image results back to client

# close socket
f.close()
# clientsocket.close()
listensocket.close()
del d
