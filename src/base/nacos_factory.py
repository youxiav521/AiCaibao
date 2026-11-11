import asyncio
from typing import Optional
from nacos import NacosClient
from src.config import Config
import socket
from loguru import logger

def getIp() -> str:
    """获取本机 IP 地址，解析失败则返回 127.0.0.1"""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip
    except socket.gaierror:
        logger.warning("无法解析主机名 {}，使用 127.0.0.1 作为回退", socket.gethostname())
        return "127.0.0.1"

class NacosManager:
    def __init__(self, config: Config):
        self.config = config
        self.serverAddr = self.config.nacos.serverAddr
        self.namespace = self.config.nacos.namespace
        self.username = self.config.nacos.username
        self.password = self.config.nacos.password
        self.appName = self.config.server.name
        self.host = self.config.nacos.instanceIp 
        self.port = self.config.nacos.instancePort
        self.group = self.config.nacos.group
        self.client = None
        self.task: Optional[asyncio.Task] = None 
        if not self.host:
            self.host = getIp()
    def register(self):
        
        """注册到nacos客户端"""
        if not self.config.nacos.enabled:
            return    
        try:
            self.client = NacosClient(
                server_addresses=self.serverAddr,
                namespace=self.namespace,
                username=self.username,
                password=self.password
            )
            self.client.add_naming_instance(self.appName, self.host, self.port,group_name=self.group)
            # 启动心跳机制
            loop = asyncio.get_event_loop()
            self.task = loop.create_task(self._startHeartbeat())
            logger.info(f"成功注册到Nacos:{self.appName}[{self.host}:{self.port}]")
        except Exception as e:
            logger.error(f"Failed to register to Nacos: {e}")
    def deregister(self):
        if not self.config.nacos.enabled:
            return
        """从Nacos中下线"""
        if not self.client:
            return False
        try:
            if self.task:
                self.task.cancel()
            self.client.remove_naming_instance(self.appName, self.host, self.port,group_name=self.group)
            logger.info(f"Service {self.appName} deregistered to Nacos successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister from Nacos: {e}")
            return False
        
    def _ping(self):
        try:
            self.client.send_heartbeat(self.appName, self.host, self.port,group_name=self.group)
            logger.debug(f"向Nacos发送心跳成功: {self.appName} ({self.group}) ({self.host}:{self.port})")
        except Exception as e:
            logger.error(f"向Nacos发送心跳失败: {str(e)}")
    async def _startHeartbeat(self):
        """异步定时任务，用于周期性地发送心跳"""
        try:
            while True:
                # 在一个单独的线程中执行同步的心跳发送操作，避免阻塞异步事件循环
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._ping)
                await asyncio.sleep(5)  # 默认5秒发送一次心跳
        except Exception as e:
            logger.error(f"循环发送心跳失败: {str(e)}")
        finally:
            logger.info("Heartbeat task finished")