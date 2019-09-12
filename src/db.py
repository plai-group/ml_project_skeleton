import atexit
import sys
import os
import socket
from pathlib import Path
from time import sleep
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from sacred import Experiment
from sacred.observers import MongoObserver
import subprocess

DATABASE_NAME     = 'vadmas_experiments'
DATABASE_USERNAME = 'vadmas.exp'
DATABASE_PASSWORD = 'sacred'
DATABASE_PORT     = '27017'
DATABASE_MACHINE  = 'headnode'
DATABASE_SERVER   = 'submit.cs.ubc.ca'

# Note: This assumes you have password-free access via public keys
REMOTE_SERVER          = 'remote.cs.ubc.ca'
REMOTE_SERVER_USERNAME = 'vadmas'

MONGO_URI = "mongodb://{}:{}@{}:{}/{}".format(DATABASE_USERNAME, DATABASE_PASSWORD, DATABASE_SERVER, DATABASE_PORT, DATABASE_NAME)
LOCAL_URI = "mongodb://{}:{}@{}:{}/{}".format(DATABASE_USERNAME, DATABASE_PASSWORD, "localhost", DATABASE_PORT, DATABASE_NAME)

def init(ex=None):
    if ex is None:
        ex = Experiment()
        ex = add_source_file(ex)  # To avoid db.py being source file
    if '--unobserved' in sys.argv:
        return ex
    if test_connection(MONGO_URI):
        ex.observers.append(MongoObserver.create(MONGO_URI, db_name=DATABASE_NAME))
    else:
        ssh_uri = open_ssh()
        ex.observers.append(MongoObserver.create(ssh_uri, db_name=DATABASE_NAME))
    return ex

def test_connection(uri, timeout=5):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout)
        client.server_info()
        return True
    except ServerSelectionTimeoutError:
        return False
    except Exception as ex:
        print(ex)
        raise

def open_ssh():
    open_port = find_open_port()
    print("Opening ssh tunnel on port:", open_port)
    p = subprocess.Popen(['ssh -N -L {}:{}:{} {}@{}'.format(open_port, DATABASE_MACHINE, DATABASE_PORT, REMOTE_SERVER_USERNAME, REMOTE_SERVER)], shell=True)
    print('ssh -N -L {}:{}:{} {}@{}'.format(open_port, DATABASE_MACHINE, DATABASE_PORT, REMOTE_SERVER_USERNAME, REMOTE_SERVER))
    ssh_mongo_uri = LOCAL_URI.replace(DATABASE_PORT, open_port)
    sleep(1)  # Give the tunnel a second to connect
    atexit.register(p.kill)
    assert test_connection(ssh_mongo_uri), "Error, SSH connection not established. URI: {}".format(ssh_mongo_uri)
    return ssh_mongo_uri

def add_source_file(ex):
    filename = Path(os.getcwd()) / sys.argv[0]
    ex.add_source_file(filename)
    return ex

def find_open_port():
    s = socket.socket()
    s.bind(('', 0))                 # Bind to a free port provided by the host.
    return str(s.getsockname()[1])  # Return the port number assigned.
