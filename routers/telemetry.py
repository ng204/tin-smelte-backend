import socket
import threading
import re
import os
from typing import Optional

import requests
from fastapi import APIRouter

# FastAPI 路由
router = APIRouter(prefix="/telemetry", tags=["遥测(温度)"])

_server_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_server_socket: Optional[socket.socket] = None
_listen_host = "0.0.0.0"
_listen_port = 8080
_forward_url: Optional[str] = None


def _get_default_forward_url() -> str:
    # 优先使用环境变量，其次尝试本机 8001，最后 8000
    env_url = os.getenv("RTM_BASE_URL")
    if env_url:
        return env_url.rstrip("/") + "/api/real-time-monitoring/update-data"
    return "http://127.0.0.1:8001/api/real-time-monitoring/update-data"


def _post_temperature(value: float, data_type: str = "crystallizer_temperature"):
    url = (_forward_url or _get_default_forward_url()).rstrip("/")
    try:
        resp = requests.post(url, json={
            "data_type": data_type,
            "value": value,
            "unit": "°C",
            "source": "factory"
        }, timeout=3)
        resp.raise_for_status()
    except Exception:
        # 忽略上报失败，避免阻塞采集
        pass


def _parse_temperature(payload: str) -> Optional[float]:
    # 尽量从字符串中提取最后一个数字（如 'uart1 data: 749' -> 749）
    matches = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)", payload)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


## 双通道支持移除（当前仅接收串口1/结晶机温度）


def _handle_client(client_socket: socket.socket, client_addr):
    try:
        client_socket.settimeout(1.0)
        buffer = b""
        while not _stop_event.is_set():
            try:
                chunk = client_socket.recv(1024)
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk
            # 按行切分
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode('utf-8', errors='replace').strip()
                if not text:
                    continue
                temp = _parse_temperature(text)
                if temp is not None:
                    # 简单区分：包含 'uart4' 视为顶吹炉温度，其它视为结晶机温度
                    data_type = "furnace_temperature" if ("uart4" in text.lower()) else "crystallizer_temperature"
                    _post_temperature(temp, data_type=data_type)
    finally:
        try:
            client_socket.close()
        except Exception:
            pass


def start_tcp_server(host="0.0.0.0", port=8080, forward_url: Optional[str] = None):
    """
    简单的TCP服务器，用于接收DNV307发来的温度数据
    :param host: 监听IP（这里写你主机的公网/内网IP）
    :param port: 监听端口（需和开发板配置一致）
    :param forward_url: 将解析的温度上报到后端实时接口的URL
    """
    global _server_socket, _listen_host, _listen_port, _forward_url

    _listen_host = host
    _listen_port = port
    _forward_url = forward_url

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    server_socket.settimeout(1.0)
    _server_socket = server_socket
    print(f"TCP服务器已启动，监听 {host}:{port}")

    try:
        while not _stop_event.is_set():
            try:
                client_socket, client_addr = server_socket.accept()
            except socket.timeout:
                continue
            print(f"新连接: {client_addr}")
            threading.Thread(target=_handle_client, args=(client_socket, client_addr), daemon=True).start()

    except KeyboardInterrupt:
        print("服务器已关闭")
    finally:
        try:
            server_socket.close()
        except Exception:
            pass


def _run_server():
    try:
        start_tcp_server(_listen_host, _listen_port, _forward_url)
    except Exception as e:
        print(f"TCP服务器运行异常: {e}")


@router.get("/status")
def telemetry_status():
    return {
        "running": _server_thread is not None and _server_thread.is_alive(),
        "host": _listen_host,
        "port": _listen_port,
        "forward_url": _forward_url or _get_default_forward_url(),
    }


@router.post("/start")
def telemetry_start(host: Optional[str] = None, port: Optional[int] = None, forward_url: Optional[str] = None):
    global _server_thread, _listen_host, _listen_port, _forward_url
    if _server_thread is not None and _server_thread.is_alive():
        return telemetry_status()
    _listen_host = host or _listen_host
    _listen_port = port or _listen_port
    _forward_url = forward_url or _forward_url
    _stop_event.clear()
    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
    return telemetry_status()


@router.post("/stop")
def telemetry_stop():
    global _server_thread
    _stop_event.set()
    try:
        if _server_socket:
            _server_socket.close()
    except Exception:
        pass
    _server_thread = None
    return {"running": False}


if __name__ == "__main__":
    # 直接运行用于本地调试
    start_tcp_server(host="0.0.0.0", port=8080)

