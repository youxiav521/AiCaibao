"""
@Time    : 2026/01/21
@Desc    : 回调工具函数
提供获取回调地址和发送回调请求的功能
"""
import os
import json
from typing import Any, Dict, Optional
import httpx
from loguru import logger

def get_callback_url() -> str:
    """
    从环境变量获取回调URL
    环境变量: CONTENT_GENERATE_CALLBACK_URL
    如果未设置，将抛出RuntimeError
    Returns:
        str: 完整的回调URL
    """
    # 从环境变量读取回调URL
    callback_url = os.getenv("CONTENT_GENERATE_CALLBACK_URL")
    if callback_url is None or callback_url.strip() == "":
        raise RuntimeError(
            "环境变量 CONTENT_GENERATE_CALLBACK_URL 未配置。"
            "请在.env文件中设置该变量，例如：CONTENT_GENERATE_CALLBACK_URL=http://example.com/api/callback"
        )
    
    # 移除末尾的斜杠（如果有）
    callback_url = callback_url.rstrip("/")
    return f"{callback_url}/api/aireporttemplate/templateCallBack"


async def send_callback(
    data: Dict[str, Any],
    template_id: str,
    event: str = "success",
    callback_url: Optional[str] = None,
    timeout: float = 30.0
) -> bool:
    """
    异步发送回调请求到指定的回调地址
    Args:
        data: 事件数据，将被序列化为JSON字符串
        template_id: 模板ID
        event: 事件类型（例如：start, success, complete, error），默认为success
        callback_url: 回调URL，如果为None则使用环境变量中的默认值
        timeout: 请求超时时间（秒），默认30秒
    Returns:
        bool: 是否需要中断当前任务
    """
    try:
        # 使用传入的回调URL，如果没有则使用环境变量中的默认值
        url = callback_url if callback_url else get_callback_url()
        
        logger.info(f"准备发送回调 - URL: {url}, 事件: {event}, 模板ID: {template_id}")
        
        # 将data序列化为JSON字符串
        data_json_str = json.dumps(data, ensure_ascii=False)
        
        # 构建回调负载
        callback_payload = {
            "data": data_json_str,
            "event": event,
            "templateId": template_id
        }
        
        # 发送异步POST请求
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=callback_payload)
            response.raise_for_status()
            
            logger.info(
                f"回调请求成功 - 状态码: {response.status_code}, "
                f"模板ID: {template_id}, 事件: {event}"
            )
            
            # 解析回调服务端返回的响应
            interrupt_payload: Optional[Dict[str, Any]] = None
            try:
                interrupt_payload = response.json().get("data")
            except Exception as e:
                logger.warning(f"回调返回JSON解析失败: {response.text}, 错误: {e}")
                return False
            
            # 检查是否需要中断任务
            if isinstance(interrupt_payload, dict):
                resp_template_id = interrupt_payload.get("templateId")
                interrupt = interrupt_payload.get("interrupt")
                
                logger.info(
                    f"回调返回数据 - 模板ID: {resp_template_id}, "
                    f"中断标志: {interrupt}"
                )
                
                # 验证返回的templateId是否匹配
                if resp_template_id != template_id:
                    logger.error(
                        f"回调返回的模板ID不匹配 - "
                        f"期望: {template_id}, 实际: {resp_template_id}"
                    )
                    return False
                
                # 检查是否收到中断指令
                if resp_template_id == template_id and interrupt is True:
                    logger.warning(
                        f"收到中断指令 - 模板ID: {template_id}, "
                        f"将停止当前任务执行"
                    )
                    return True
            
            return False
            
    except httpx.HTTPStatusError as e:
        logger.error(
            f"回调请求HTTP错误 - "
            f"状态码: {e.response.status_code}, "
            f"URL: {url}, 模板ID: {template_id}, 事件: {event}, "
            f"错误: {e}"
        )
    except httpx.TimeoutException as e:
        logger.error(
            f"回调请求超时 - "
            f"URL: {url}, 模板ID: {template_id}, "
            f"超时时间: {timeout}秒, 错误: {e}"
        )
    except httpx.RequestError as e:
        logger.error(
            f"回调请求失败 - "
            f"URL: {url}, 模板ID: {template_id}, 事件: {event}, "
            f"错误: {e}"
        )
    except Exception as e:
        logger.error(
            f"回调过程中发生未知错误 - "
            f"URL: {url}, 模板ID: {template_id}, 事件: {event}, "
            f"错误类型: {type(e).__name__}, 错误: {e}",
            exc_info=True
        )
    
    return False


async def send_processing_callback(
    data: Dict[str, Any],
    template_id: str,
    callback_url: Optional[str] = None
) -> bool:
    """
    发送处理进度回调的便捷函数
    Args:
        data: 处理数据（例如：{"id": "q_1", "description": "...", "type": "text"}）
        template_id: 模板ID
        callback_url: 可选的回调URL
    Returns:
        bool: 是否需要中断任务
    """
    return await send_callback(
        data=data,
        template_id=template_id,
        event="success",
        callback_url=callback_url
    )


async def send_start_callback(
    data: Dict[str, Any],
    template_id: str,
    callback_url: Optional[str] = None
) -> bool:
    """
    发送任务开始回调的便捷函数
    Args:
        data: 开始数据（例如：{"total": 75, "message": "开始处理"}）
        template_id: 模板ID
        callback_url: 可选的回调URL
    Returns:
        bool: 是否需要中断任务
    """
    return await send_callback(
        data=data,
        template_id=template_id,
        event="start",
        callback_url=callback_url
    )


async def send_completed_callback(
    data: Dict[str, Any],
    template_id: str,
    callback_url: Optional[str] = None
) -> bool:
    """
    发送任务完成回调的便捷函数
    Args:
        data: 完成数据（例如：{"total": 75, "success": True}）
        template_id: 模板ID
        callback_url: 可选的回调URL
    Returns:
        bool: 是否需要中断任务（通常完成后不需要）
    """
    return await send_callback(
        data=data,
        template_id=template_id,
        event="complete",
        callback_url=callback_url
    )


async def send_error_callback(
    data: Dict[str, Any],
    template_id: str,
    callback_url: Optional[str] = None
) -> bool:
    """
    发送错误回调的便捷函数
    Args:
        data: 错误数据（例如：{"error": "处理失败", "detail": "..."}）
        template_id: 模板ID
        callback_url: 可选的回调URL
    Returns:
        bool: 是否需要中断任务
    """
    return await send_callback(
        data=data,
        template_id=template_id,
        event="error",
        callback_url=callback_url
    )


if __name__ == "__main__":
    import asyncio
    
    async def test():
        """测试回调函数"""
        print("=" * 60)
        print("测试回调工具函数")
        print("=" * 60)
        
        # 设置测试环境变量
        os.environ["CONTENT_GENERATE_CALLBACK_URL"] = "http://localhost:8000/api/callback"
        
        # 测试1: 获取回调URL
        print("\n测试1: 获取回调URL")
        try:
            url = get_callback_url()
            print(f"✓ 回调URL: {url}")
        except Exception as e:
            print(f"✗ 获取回调URL失败: {e}")
        
        # 测试2: 发送开始回调
        print("\n测试2: 发送开始回调")
        try:
            interrupted = await send_start_callback(
                data={"total": 10, "message": "开始处理"},
                template_id="test_template_123"
            )
            print(f"✓ 回调发送完成，中断标志: {interrupted}")
        except Exception as e:
            print(f"✗ 发送回调失败: {e}")
        
        # 测试3: 发送处理进度回调
        print("\n测试3: 发送处理进度回调")
        try:
            interrupted = await send_processing_callback(
                data={"id": "q_5", "description": "填写公司名称", "type": "text"},
                template_id="test_template_123"
            )
            print(f"✓ 回调发送完成，中断标志: {interrupted}")
        except Exception as e:
            print(f"✗ 发送回调失败: {e}")
        
        # 测试4: 发送完成回调
        print("\n测试4: 发送完成回调")
        try:
            interrupted = await send_completed_callback(
                data={"total": 10, "processed": 10, "success": True},
                template_id="test_template_123"
            )
            print(f"✓ 完成回调发送成功，中断标志: {interrupted}")
        except Exception as e:
            print(f"✗ 发送完成回调失败: {e}")
        
        # 测试5: 发送错误回调
        print("\n测试5: 发送错误回调")
        try:
            interrupted = await send_error_callback(
                data={"error": "处理失败", "detail": "网络连接超时"},
                template_id="test_template_123"
            )
            print(f"✓ 错误回调发送成功，中断标志: {interrupted}")
        except Exception as e:
            print(f"✗ 发送错误回调失败: {e}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
    
    asyncio.run(test())

