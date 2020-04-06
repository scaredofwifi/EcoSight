# The socket server to communicate with the Android Studio Application and read in the images taken by the user

import socket
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

# accepts incoming connection
(clientsocket, address) = listensocket.accept()
print("New connection made!")

# open file
f = open('incoming.jpg', 'wb')  # TODO : change the filepath of this incoming image
datain = 1

# receive image
while datain:
    datain = clientsocket.recv(1024)  # get incoming data
    if not datain:
        break
    f.write(datain)  # writes data to file

print("received image!")
# process image
# process.imgpreprocessing(data)

# TODO : send image results back to client

# close socket
f.close()
# clientsocket.close()
listensocket.close()