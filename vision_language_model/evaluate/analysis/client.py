# client.py (trong môi trường khác)
from multiprocessing.connection import Client
import json

def send_prompt(prompt, temperature=0.7, top_p=0.8, top_k=None, repetition_penalty=1.05, max_tokens=512):
    # Kết nối tới server
    conn = Client(('localhost', 6000), authkey=b'secret_password')
    
    # Gửi prompt và các tham số
    data = {
        "messages": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens
    }
    conn.send(data)

    # Nhận kết quả
    response = conn.recv()
    
    # Đóng kết nối
    conn.send("close")
    conn.close()
    
    return response["response"]

if __name__ == "__main__":
    # Ví dụ sử dụng
    prompt = "Tell me something about large language models."
    result = send_prompt(prompt, top_k=50)
    print("Response:", result)