# server.py (tối ưu cho 2 GPU, batching, nhiều client)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from multiprocessing.connection import Listener
import threading
import queue
import time
import logging

logging.basicConfig(level=logging.INFO)

def handle_client(conn, request_queue):
    """Nhận yêu cầu từ client và gửi vào queue."""
    try:
        while True:
            data = conn.recv()
            if data == "close":
                break
            request_queue.put((conn, data))
    except EOFError:
        logging.info("Connection closed by client")
    finally:
        conn.close()

def process_requests(request_queue, llm, tokenizer, response_queue):
    """Xử lý batch các yêu cầu từ queue."""
    while True:
        batch = []
        start_time = time.time()
        # Thu thập tối đa 4 yêu cầu hoặc chờ 0.5 giây
        while len(batch) < 4 and (time.time() - start_time) < 0.5:
            try:
                conn, data = request_queue.get_nowait()
                batch.append((conn, data))
            except queue.Empty:
                time.sleep(0.01)

        if not batch:
            continue

        logging.info(f"Processing batch of {len(batch)} prompts")
        texts = []
        sampling_params_list = []
        conns = []
        for conn, data in batch:
            prompt = data.get("messages", "")  # Giữ key "messages" như code chuẩn
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 1.0)  # Mặc định giống OpenAI
            top_k = data.get("top_k", -1)
            repetition_penalty = data.get("repetition_penalty", 1.05)
            max_tokens = data.get("max_tokens", 256)  # Giảm để tăng tốc

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            )
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
            sampling_params_list.append(sampling_params)
            conns.append(conn)

        # Xử lý batch
        outputs = llm.generate(texts, sampling_params_list[0])  # Dùng params đầu tiên
        for (conn, _), output in zip(batch, outputs):
            response_queue.put((conn, {"response": output.outputs[0].text}))

def process_responses(response_queue):
    """Gửi phản hồi về client."""
    while True:
        conn, response = response_queue.get()
        try:
            conn.send(response)
        except Exception as e:
            logging.error(f"Error sending response: {e}")
        response_queue.task_done()

def start_server(model_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,6"
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=8096,  # Giảm để tăng tốc
        enable_prefix_caching=True,
        tensor_parallel_size=2,  # Khớp với CUDA_VISIBLE_DEVICES="4,6"
        distributed_executor_backend="mp",  # Tắt Ray,
        dtype="bfloat16",
        
    )
    logging.info("Model loaded successfully!")

    request_queue = queue.Queue()
    response_queue = queue.Queue()

    # Luồng xử lý batch
    batch_thread = threading.Thread(
        target=process_requests,
        args=(request_queue, llm, tokenizer, response_queue),
        daemon=True
    )
    batch_thread.start()

    # Luồng gửi phản hồi
    response_thread = threading.Thread(
        target=process_responses,
        args=(response_queue,),
        daemon=True
    )
    response_thread.start()

    # Khởi tạo server
    listener = Listener(('localhost', 6000), authkey=b'secret_password')
    logging.info("Server started, waiting for connections...")

    while True:
        conn = listener.accept()
        logging.info(f"Connection accepted from {listener.last_accepted}")
        client_thread = threading.Thread(
            target=handle_client,
            args=(conn, request_queue),
            daemon=True
        )
        client_thread.start()

if __name__ == "__main__":
    model_path = "/cm/archive/anonymous/checkpoints/benchmarks/DeepSeek-R1-Distill-Qwen-32B"
    start_server(model_path)