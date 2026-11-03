import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import traceback
import unittest
from unittest.mock import patch

from emergent_lab.transport import GenerationCancelled, stream_json


class TransportCancellationTests(unittest.TestCase):
    def check_cancel_before_first_chunk(self, protocol, chunked=False):
        release_server = threading.Event()
        entered_read = threading.Event()
        connection_released_socket = threading.Event()
        outcome, chunks = [], []

        class StalledHandler(BaseHTTPRequestHandler):
            protocol_version = protocol

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.flush()
                # Simulate a model still reasoning: no generation chunk exists
                # until well after the client cancellation deadline.
                release_server.wait(8)
                self.close_connection = True

            def log_message(self, *args):
                pass

        class ObservedResponse(http.client.HTTPResponse):
            def readline(self, limit=-1):
                entered_read.set()
                return super().readline(limit)

        class ObservedConnection(http.client.HTTPConnection):
            response_class = ObservedResponse

            def getresponse(self):
                response = super().getresponse()
                if self.sock is None:
                    connection_released_socket.set()
                return response

        server = ThreadingHTTPServer(("127.0.0.1", 0), StalledHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        cancel = threading.Event()
        port = server.server_address[1]

        def run_client():
            try:
                stream_json(f"http://127.0.0.1:{port}", "/generate", {}, chunks.append,
                            cancel, timeout=10)
            except BaseException as exc:
                outcome.append(exc)

        client = threading.Thread(target=run_client, daemon=True)
        try:
            connection = ObservedConnection("127.0.0.1", port, timeout=10)
            # Keep actual HTTP and sockets; only observe when the response starts
            # reading so cancellation cannot happen before headers arrive.
            with patch("emergent_lab.transport._connect", return_value=connection):
                client.start()
                self.assertTrue(entered_read.wait(2), "The client never entered its first chunk read")
                if protocol == "HTTP/1.0":
                    self.assertTrue(connection_released_socket.is_set(),
                                    "Fixture must exercise HTTPConnection's socket handoff")
                started = time.monotonic()
                cancel.set()
                client.join(2)
                elapsed = time.monotonic() - started
                self.assertFalse(client.is_alive(),
                                 "Cancellation waited for response content instead of interrupting its socket")
                self.assertLess(elapsed, 2)
            self.assertEqual(chunks, [])
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], GenerationCancelled,
                                  "".join(traceback.format_exception(outcome[0])))
        finally:
            cancel.set()
            release_server.set()
            client.join(2)
            server.shutdown()
            server.server_close()
            server_thread.join(2)

    def test_cancel_interrupts_http10_close_delimited_reply_before_any_chunk(self):
        self.check_cancel_before_first_chunk("HTTP/1.0")

    def test_cancel_interrupts_http11_chunked_reply_before_any_chunk(self):
        self.check_cancel_before_first_chunk("HTTP/1.1", chunked=True)


if __name__ == "__main__":
    unittest.main()
