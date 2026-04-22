import socket
from typing import Optional

from loguru import logger


def get_wifi_ip() -> Optional[str]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to actually connect — just forces the OS
        # to pick the outbound interface for that destination
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError as e:
        logger.warning(f"Could not determine Wi-Fi IP: {e}")
        return None
    finally:
        s.close()
