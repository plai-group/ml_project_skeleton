import atexit
import sys
import os
import socket
import uuid
from pathlib import Path
from time import sleep
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from sacred import Experiment
from sacred.observers import MongoObserver

DATABASE_NAME     = 'vadmas_experiments'
DATABASE_USERNAME = 'vadmas.exp'
DATABASE_PASSWORD = 'sacred'
DATABASE_PORT     = '27017'
DATABASE_SERVER   = 'headnode'

REMOTE_SERVER = 'remote.cs.ubc.ca'
REMOTE_SERVER_USERNAME = 'vadmas'
REMOTE_SERVER_PASSWORD = None  # Assumes you have password-free access via public keys
REMOTE_MONGO_URI = "mongodb://{}:{}@{}:{}/{}".format(DATABASE_USERNAME,
                                                     DATABASE_PASSWORD,
                                                     REMOTE_SERVER,
                                                     DATABASE_PORT,
                                                     DATABASE_NAME)
SSH_CONTROL_PATH = Path('./.ssh_control_path')
SSH_SESSION = "%r@%h:%p_{}".format(uuid.uuid1())


# Better to pass in ex from the main file to
# use sacred's automatic source code discovery
# functionality

def init(ex=None):
    if ex is None:
        ex = Experiment()
        ex = add_source_file(ex)  # To avoid db.py being source file
    if '--unobserved' in sys.argv:
        return ex
    if test_connection(REMOTE_MONGO_URI):
        ex.observers.append(MongoObserver.create(REMOTE_MONGO_URI, db_name=DATABASE_NAME))
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
    SSH_CONTROL_PATH.mkdir(parents=True, exist_ok=True)
    os.system('ssh -f -N -M -S {}/{} -L {}:{}:27017 {}@{}'.format(SSH_CONTROL_PATH,
                                                                  SSH_SESSION,
                                                                  open_port,
                                                                  DATABASE_SERVER,
                                                                  REMOTE_SERVER_USERNAME,
                                                                  REMOTE_SERVER))
    atexit.register(close_ssh)
    ssh_mongo_uri = REMOTE_MONGO_URI.replace(REMOTE_SERVER, "localhost").replace(DATABASE_PORT, open_port)
    sleep(1)  # Give the tunnel a second to connect
    assert test_connection(ssh_mongo_uri), "Error, SSH connection not established"
    return ssh_mongo_uri


def add_source_file(ex):
    filename = Path(os.getcwd()) / sys.argv[0]
    ex.add_source_file(filename)
    return ex


def find_open_port():
    s = socket.socket()
    s.bind(('', 0))                 # Bind to a free port provided by the host.
    return str(s.getsockname()[1])  # Return the port number assigned.


def close_ssh():
    os.system('ssh -S {}/{} -O exit {}@{}'.format(SSH_CONTROL_PATH, SSH_SESSION, REMOTE_SERVER_USERNAME, REMOTE_SERVER))
    print("Closing ssh tunnel.")
