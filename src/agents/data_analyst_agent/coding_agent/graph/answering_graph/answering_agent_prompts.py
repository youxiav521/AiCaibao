"""
answering_agent提示词存放位置
"""

answer_user_prompt = """
以下是用户的问题，请根据回答思路和{related_data_label}，给出问题的回答，不需要进行任何数据的计算

<原始用户上下文数据>
{context_str}
</原始用户上下文数据>

<{related_data_tag}>
{related_data_calculation_results}
</{related_data_tag}>

<回答思路>
{answer_cot}
</回答思路>

<用户问题>
{user_question}
{user_prompt}
</用户问题>

### 回答注意事项
- 只使用<{related_data_tag}></{related_data_tag}>中提供的的数据
- 不需要从用户问题中提取任何数据或者做任何计算，只需要使用{related_data_label}中的数据，按照用户的要求回答问题
- 不要虚构任何数据，所有数据都必须是{related_data_label}中的数据。
- 如果{related_data_label}中没有对应的数据，且上下文中没有对应的数据，不要提及{related_data_label}，直接说明缺少信息无法回答，不要虚构数据进行回答
- 不要自己进行数据计算，数据转换或者数据单位转换
- 当数据没有单位时，不要假定单位，{related_data_label}的单位都是用户要求的单位
- 回答近几年或者近几期的相关数据时，应当说明时间顺序，例如：近三年资产负债率为（2021年-2023年）：78.78%、87.87%、86.69%
"""
