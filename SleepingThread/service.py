# -*- coding: utf-8 -*-

"""
Service functions, add to SleepingThread?
getRAMUsage

and IPC: PointToPoint through Sockets functions:
    SocketClient
    SocketServer
"""

def getRAMUsage(unit="MB"):
    statfile = open("/proc/self/status","r")
    for line in statfile:
        if line[0:6]=="VmRSS:":
            break

    statfile.close()

    line = line.strip().split()
    memory = 0.0
    if line[2]=='kB':
        memory = int(line[1])*1024.0
    else:
        memory = -1
        raise ValueError('/proc/self/status VmRSS not int kB')
        return -1

    if unit=="KB":
        memory = memory/1024.0
    elif unit=="MB":
        memory = memory/1024.0/1024.0
    elif unit=="GB":
        memory = memory/1024.0/1024.0

    return memory

"""
import sys
import time
import multiprocessing as mproc
from ctypes import c_char_p

def threadreader(stdin,strcommand,lock):
    while True:
        #lock.acquire()
        line = stdin.readline()
        #lock.release()
        if True:
            print line
        if strcommand.value=="stop":
            break
        time.sleep(1)
    return

class CommandReader:
    def __init__(self,stdin,timeout=1.0):
        self.manager = mproc.Manager()
        self.lock = mproc.Lock()
        self.strcommand = self.manager.Value(c_char_p,"")
        self.thr = mproc.Process(target=threadreader,args=(stdin,self.strcommand,self.lock))
        self.thr.start()
        time.sleep(timeout)
        self.thr.terminate()
        return
"""
import socket
import dill as pickle
import time

class SocketDataManager(object):
    def __init__(self):
        self.brokenconnection = False
        pass

    def readData(self,blocking=False):
        conn = self.conn
        iscommand = None
        try:
            if not blocking:
                conn.setblocking(0)
            datasize = conn.recv(16)
            datasize = int(datasize)
            if datasize<0:
                iscommand = True
                datasize = -datasize
            else:
                iscommand = False
            conn.setblocking(1)
            if datasize!=0:
                data = conn.recv(datasize)
            else:
                data = ""
            self.brokenconnection = False
        except socket.error:
            self.brokenconnection = False
            return (None,iscommand)
        except:
            self.brokenconnection = True
            return (None,iscommand)

        return (data,iscommand)

    def sendCommand(self,commandstr):
        commsize = -len(commandstr)
        try:
          self.conn.send("%016d"%commsize)
          self.conn.send(commandstr)
        except:
          self.brokenconnection = True
        return

    def sendData(self,data):
        datasize = len(data)
        try:
            self.conn.send("%016d"%datasize)
            self.conn.send(data)
        except socket.error:
            pass
        except:
            print "SocketDataManager::sendData: undefined error"

        return
    
    def readObject(self,blocking=False):
        datastr,iscommand = self.readData(blocking)
        if datastr is None:
            return (None,iscommand)
        elif iscommand:
            return (datastr,iscommand)

        obj = pickle.loads(datastr)

        return (obj,iscommand)

    def sendObject(self,obj):
        objstr = pickle.dumps(obj)
        self.sendData(objstr)
        return

    
class SocketServer(SocketDataManager):
    def __init__(self,host='localhost',port=9090):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            sock.bind((host,port))
        except socket.error:
            print "Port 9090 used now"
            raise ValueError("USED NOW")
        self.sock = sock
        super(SocketServer,self).__init__()
        return

    def getConnection(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        return

    def close(self):
        self.conn.close()
        self.sock.close()
        return

class MultiDataManager(object):
    def __init__(self):
        self.brokenconnection = False
        self.conn = []
        self.addr = []
        self.group = []
        self.curconn = 0
        pass

    def readData(self,blocking=False):
        conn = self.conn[self.curconn]
        iscommand = None
        try:
            if not blocking:
                conn.setblocking(0)
            datasize = conn.recv(16)
            datasize = int(datasize)
            if datasize<0:
                iscommand = True
                datasize = -datasize
            else:
                iscommand = False
            conn.setblocking(1)
            if datasize!=0:
                data = conn.recv(datasize)
            self.brokenconnection = False
        except socket.error:
            self.brokenconnection = False
            return (None,iscommand)
        except:
            self.brokenconnection = True
            return (None,iscommand)

        return (data,iscommand)

    def sendCommand(self,commandstr):
        commsize = -len(commandstr)
        try:
          self.conn.send("%016d"%commsize)
          self.conn.send(commandstr)
        except:
          self.brokenconnection = True
        return

    def sendData(self,data):
        datasize = len(data)
        conn = self.conn[self.curconn]
        try:
            conn.send("%016d"%datasize)
            conn.send(data)
        except socket.error:
            del self.conn[self.curconn]
            del self.addr[self.curconn]
            del self.group[self.curconn]
        except:
            print "MultiDataManager::sendData: undefined error"

        return
    
    def readObject(self,blocking=False):
        datastr,iscommand = self.readData(blocking)
        if datastr is None:
            return (None,iscommand)
        elif iscommand:
            return (datastr,iscommand)

        obj = pickle.loads(datastr)

        return (obj,iscommand)

    def sendObject(self,obj):
        objstr = pickle.dumps(obj)
        self.sendData(objstr)
        return

 

class MultiServer(MultiDataManager):
    def __init__(self,host='localhost',port=10001):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            sock.bind((host,port))
        except socket.error:
            print "Port 10001 used now"
            raise ValueError("USED NOW")
        sock.listen(1)
        self.sock = sock
        super(MultiServer,self).__init__()
        return

    def readData(self,group=None):
        if len(self.conn)<=0:
            return []
        res = []
        self.curconn=0
        counter = len(self.conn)
        while True:
            self.brokenconnection = False
            if group is not None:
                if self.group[self.curconn]!=group:
                    self.curconn = (self.curconn+1)%len(self.conn)
                    continue
            data,isobject = super(MultiServer,self).readData()
            counter -= 1
            if self.brokenconnection:
                #delete conn and addr
                del self.conn[self.curconn]
                del self.addr[self.curconn]
                del self.group[self.curconn]
                if len(self.conn)==0:
                    return []
                self.curconn = (self.curconn)%len(self.conn)
            else:
                self.curconn = (self.curconn+1)%len(self.conn)
                res.append((data,isobject))
            
            if counter==0:
                break
        
        return res

    def sendData(self,data,group):
        self.curconn = 0
        while self.curconn<len(self.conn):
            if self.group[self.curconn]==group:
                super(MultiServer,self).sendData(data)
            self.curconn = self.curconn+1
        self.curconn = 0
        return

    def getConnection(self):
        breakconn = False
        while not breakconn:
            sock = self.sock
            sock.setblocking(0)
            try:
                conn,addr = sock.accept()
                self.conn.append(conn)
                self.addr.append(addr)
                #client send it's group
                sock.setblocking(1)
                self.group.append(int(conn.recv(1)))
            except:
                breakconn = True
            sock.setblocking(1)
        
        return

    def close(self):
        for conn in self.conn:
            conn.close()
        self.sock.close()
        return

class CommandSocketServer(SocketDataManager):
    def __init__(self,host='localhost',port=9091,maxcmds=1):
        if maxcmds>1:
            raise ValueError("maxcmds must be 1")
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind((host,port))
        self.sock = sock
        self.curcmds = 0
        self.maxcmds = maxcmds
        super(CommandSocketServer,self).__init__()
        return

    def getCommand(self):
        if self.curcmds==0:
            return None
        obj,iscommand = self.readObject()
        if self.brokenconnection:
            self.curcmds = 0
            return None
        if obj is not None:
            if iscommand:
                #obj - command string
                return obj
        return None

    def getConnection(self):
        if self.curcmds>=self.maxcmds:
            return
        try:
            self.sock.settimeout(0.1)
            self.sock.listen(1)
            self.conn, self.addr = self.sock.accept()
            self.curcmds += 1
        except:
            pass

        return

    def close(self):
        self.conn.close()
        self.sock.close()
        return

class SocketClient(SocketDataManager):
    def __init__(self,host='localhost',port=9090):
        self.host = host
        self.port = port

        sock = socket.socket()
        self.sock = sock
        super(SocketClient,self).__init__()
        return

    def sendGroup(self,group):
        self.sock.send(str(group))
        return

    def getConnection(self,timeout=0.1,attempts=2):
        for i in range(0,attempts):
            try:
                sock = self.sock
                sock.connect((self.host,self.port))
                self.conn = self.sock
                return True
            except socket.error as err:
                print err
            if i<attempts-1:
                time.sleep(timeout)
        return False

    def close(self):
        self.sock.close()
        return

class CommandSocketClient(SocketClient):
    def __init__(self,host='localhost',port=9091):
        super(CommandSocketClient,self).__init__(host=host,port=port)
        return
