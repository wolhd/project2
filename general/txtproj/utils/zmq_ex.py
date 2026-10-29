import queue
import threading
import time
import zmq

# Import the generated protobuf module
import sensor_pb2


def zmq_receiver(sub_endpoint, topic, message_queue):
  """Background thread to read incoming messages from ZMQ SUB socket."""
  ctx = zmq.Context.instance()
  sub_socket = ctx.socket(zmq.SUB)
  sub_socket.connect(sub_endpoint)
  sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

  print(f"[Receiver] Subscribed to {sub_endpoint} on topic '{topic}'")

  try:
    while True:
      # Receive multipart message: [topic, payload]
      parts = sub_socket.recv_multipart()
      if len(parts) == 2:
        _, payload = parts
        msg = sensor_pb2.InputMessage()
        msg.ParseFromString(payload)
        message_queue.put(msg)
  except zmq.ContextTerminated:
    pass
  finally:
    sub_socket.close()


def worker_and_publisher(pub_endpoint, message_queue):
  """Worker thread that pulls from queue, does work, and publishes results."""
  ctx = zmq.Context.instance()
  pub_socket = ctx.socket(zmq.PUB)
  pub_socket.bind(pub_endpoint)

  print(f"[Worker] Publishing processed messages on {pub_endpoint}")
  # Allow subscribers time to connect
  time.sleep(0.5)

  try:
    while True:
      msg = message_queue.get()
      if msg is None:
        break

      # --- Do Some Work ---
      print(f"[Worker] Processing ID: {msg.id}, Value: {msg.value}")
      time.sleep(0.1)  # Simulate workload

      # Create output protobuf message
      out_msg = sensor_pb2.OutputMessage()
      out_msg.id = msg.id
      out_msg.timestamp = time.time()
      out_msg.processed_value = msg.value * 2.0  # Example transformation
      out_msg.status = "SUCCESS"

      # Serialize and publish
      payload = out_msg.SerializeToString()
      pub_socket.send_multipart([b"processed.sensor", payload])

      message_queue.task_done()
  except zmq.ContextTerminated:
    pass
  finally:
    pub_socket.close()


def main():
  SUB_ENDPOINT = "tcp://localhost:5555"
  PUB_ENDPOINT = "tcp://localhost:5556"
  TOPIC = "sensor.data"

  q = queue.Queue()

  # Start receiver thread
  receiver_thread = threading.Thread(
      target=zmq_receiver, args=(SUB_ENDPOINT, TOPIC, q), daemon=True
  )
  receiver_thread.start()

  # Start worker/publisher thread
  worker_thread = threading.Thread(
      target=worker_and_publisher, args=(PUB_ENDPOINT, q), daemon=True
  )
  worker_thread.start()

  print("[Main] Service is running. Press Ctrl+C to exit.")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    print("[Main] Shutting down...")


if __name__ == "__main__":
  main()
#---
syntax = "proto3";
package app;

message InputMessage {
    string id = 1;
    double timestamp = 2;
    double value = 3;
}

message OutputMessage {
    string id = 1;
    double timestamp = 2;
    double processed_value = 3;
    string status = 4;
}
#---
# test cient
import time
import zmq

import sensor_pb2


def main():
  PUB_ENDPOINT = "tcp://*:5555"
  SUB_ENDPOINT = "tcp://localhost:5556"
  TOPIC = "sensor.data"

  ctx = zmq.Context()

  # Setup publisher to send data to the main program
  pub_socket = ctx.socket(zmq.PUB)
  pub_socket.bind(PUB_ENDPOINT)

  # Setup subscriber to listen to the processed output from the main program
  sub_socket = ctx.socket(zmq.SUB)
  sub_socket.connect(SUB_ENDPOINT)
  sub_socket.setsockopt_string(zmq.SUBSCRIBE, "processed.sensor")

  print(f"[Client] Publishing messages to {PUB_ENDPOINT}")
  print(f"[Client] Listening for processed results on {SUB_ENDPOINT}")

  # Give ZMQ connections a moment to establish
  time.sleep(0.5)

  msg_id = 1
  try:
    while True:
      # 1. Create and send input protobuf message
      in_msg = sensor_pb2.InputMessage()
      in_msg.id = f"device-{msg_id:03d}"
      in_msg.timestamp = time.time()
      in_msg.value = float(msg_id * 10)

      payload = in_msg.SerializeToString()
      pub_socket.send_multipart([TOPIC.encode("utf-8"), payload])
      print(f"[Client Sent] ID: {in_msg.id}, Value: {in_msg.value}")

      # 2. Check if there are any processed responses available (non-blocking poll)
      if sub_socket.poll(timeout=200):
        parts = sub_socket.recv_multipart()
        if len(parts) == 2:
          _, reply_payload = parts
          out_msg = sensor_pb2.OutputMessage()
          out_msg.ParseFromString(reply_payload)
          print(
              f"  [Client Received] ID: {out_msg.id}, Processed Value:"
              f" {out_msg.processed_value}, Status: {out_msg.status}"
          )

      msg_id += 1
      time.sleep(1.0)

  except KeyboardInterrupt:
    print("[Client] Stopping test client...")
  finally:
    ctx.term()


if __name__ == "__main__":
  main()
