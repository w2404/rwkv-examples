"""
一个尽可能简单的服务器，作为model的包装，不添加任何修饰。

因为目标是模仿chatgpt的api，所以没有做成对话的必要，因为chatgpt的api不是对话性的
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
#import nochatback
import chat_simple

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def do_POST(self):
        length = int(self.headers.get('Content-Length'))
        message = self.rfile.read(length).decode()
        self._set_headers()
        for s in chat_simple.on_message(message):
            self.wfile.write(s.encode())

httpd = HTTPServer(('', 8008), Server)
httpd.serve_forever()
