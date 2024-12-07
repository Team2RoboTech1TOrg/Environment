from app import run
from app_server import run_server

server = True

if __name__ == '__main__':
    run_server() if server else run()
