# The socket server to communicate with the Android Studio Application and read in the images taken by the user

import socket
import train_test as tt
import time
import preprocessing as process

# set up server listener
listensocket = socket.socket()
Port = 8000
maxConnections = 999
IP = socket.gethostname()  # get hostname of current machine

listensocket.bind((IP, Port))

# start server
listensocket.listen(maxConnections)
print("Server started at " + IP + " on port " + str(Port))


print("[status] classification model started")
# accepts incoming connection
(clientsocket, address) = listensocket.accept()
print("New connection made!")
d = Demo()

# open file
 f = open('incoming.jpg', 'wb')
data_in = 1

to_count = 0

while data_in:
    data_in = clientsocket.recv(1024).decode()
    if not data_in:
        time.sleep(3)
        if to_count == 10:
            print("Socket connection timeout. Closing connection.")
            break
        else:
            count = count + 1
    f.write(data_in)
    id_pred = tt.classify('incoming.jpg')
    clientsocket.recv(id_pred.encode())


# close socket
clientsocket.close()
listensocket.close()
del d

