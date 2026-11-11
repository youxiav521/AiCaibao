"""
@Time    : 2026/01/21
@Desc    : 提示词优化工具
通过LLM生成优化后的提示词变体
"""
from typing import List

from src.agents.content_generate_agent.utils import abatch_llm_with_prompt
from src.agents.template.optimize_prompt.prompt import get_optimization_prompt
from loguru import logger


async def optimize_prompt(
    original_prompt: str,
    num_variants: int = 1,
    model_id: str = "model_bailian_80b",
    optimization_focus: str = "清晰度、完整性和效果",
    use_english: bool = False
) -> List[str]:
    """
    优化单个提示词，生成多个优化变体
    每次生成一个优化版本，使用批量调用提高效率
    Args:
        original_prompt: 原始提示词
        num_variants: 需要生成的优化变体数量，默认3个
        model_id: 使用的模型ID
        optimization_focus: 优化重点，默认关注"清晰度、完整性和效果"
        use_english: 是否使用英文版本
    Returns:
        优化后的提示词列表
    """
    if num_variants < 1:
        raise ValueError("num_variants 必须大于等于 1")
    
    logger.info(f"开始优化提示词，生成 {num_variants} 个变体...")
    
    # 定义不同版本的侧重点
    variant_focuses = [
        "清晰度和可理解性",
        "完整性和结构化",
        "效果和实用性",
        "简洁性和效率",
        "创新性和多样性"
    ]
    
    # 构建多个优化提示词，每个要求生成一个版本
    optimization_prompts = []
    for i in range(num_variants):
        focus = variant_focuses[i % len(variant_focuses)]
        
        optimization_prompt = get_optimization_prompt(
            optimization_focus=optimization_focus,
            focus=focus,
            original_prompt=original_prompt,
            use_english=use_english
        )
        optimization_prompts.append(optimization_prompt)
    
    logger.info(f"构建的优化提示词: {original_prompt}")
    try:
        # 使用批量调用方法
        responses = await abatch_llm_with_prompt(
            prompts=optimization_prompts,
            model_id=model_id,
        )
        
        # 解析返回结果，提取优化后的提示词
        optimized_prompts = []
        for idx, response in enumerate(responses):
            # 直接使用返回的文本
            optimized_prompts.append(response.strip())
        
        # 过滤空结果
        optimized_prompts = [p for p in optimized_prompts if p]
        
        # 确保返回的数量正确
        if len(optimized_prompts) != num_variants:
            logger.warning(
                f"生成的提示词数量 ({len(optimized_prompts)}) "
                f"与请求数量 ({num_variants}) 不符"
            )
        
        logger.info(f"成功生成 {len(optimized_prompts)} 个优化提示词")
        return optimized_prompts[:num_variants]  # 确保不超过请求的数量
        
    except Exception as e:
        logger.error(f"提示词优化失败: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    
    async def test():
        # 测试单个提示词优化
        original = "请帮我分析这段文本的情感"
        
        print("=" * 50)
        print("测试: 生成3个优化版本")
        print("=" * 50)
        optimized = await optimize_prompt(original, num_variants=3)
        for i, prompt in enumerate(optimized, 1):
            print(f"\n优化版本 {i}:")
            print(prompt)
        
        print("\n" + "=" * 50)
        print("测试: 英文版本")
        print("=" * 50)
        optimized_en = await optimize_prompt(original, num_variants=1, use_english=True)
        print(f"\n英文优化版本:")
        print(optimized_en[0])
    
    asyncio.run(test())
