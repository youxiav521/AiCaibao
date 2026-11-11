"""
coding_agent提示词存放位置
"""

plan_text = """
## 问题描述：
{question_description}
{user_prompt}

## 解决思路：
### 回答要点
{answer_keypoints}
### 需要计算的指标以及限制条件
```json
{need_calculate_metrics}
```
"""

input_question_prompt = """
以下是用户包含上下文的问题，注意准确理解问题需要的计算，并正确计算，注意不要虚构任何数据：
# 上下文信息
相关的上下文信息如下:
{context}
# 可读取的表格名
{table_info}
# 用户输入
{plan_text}

## 任务要求
- 你的最终目的是判断回答用户问题需要的指标计算，并使用代码准确计算回答用户问题需要的指标，在打印结果时中按照用户要求保留小数点精度
- 你不是要直接回答用户的问题，而是调用工具`execute_code_tool`来执行代码，并确保代码的执行结果符合问题要求
- 任何你期望从代码中返回的结果，都要在代码中使用print()函数打印以支持返回，包含计算结果等
- 你的代码中导入了预先定义好加减乘除和保留指定小数位的方法`add(a,b,scale),sub(a,b,scale),mul(a,b,scale),div(a,b,scale),quantize(value,scale)`，无需再次导入直接使用即可
- 严禁在代码中直接使用 +、-、*、/ 运算符进行任何数值计算，代码中只要涉及数值的加、减、乘、除，必须调用这些函数来进行计算
- 读取表格在代码中使用预先定义的`read_table(table_name:str)`，方法返回pandas.Dataframe类型的表格数据
- `read_table(table_name:str)`方法无需再次导入，直接使用方法来读取对应<可读取的表格名>中存在的表格数据
- 不要在代码中使用`read_table(table_name:str)`方法读取<可读取的表格名>中不存在的表格名，当<可读取的表格名>中没有表格数据时，不要尝试使用`read_table(table_name:str)`方法读取表格数据
- 代码中使用'print()'来打印所有的结果，所有期望代码中返回的结果都要通过'print()'来打印
- 打印结果时，如果有对应的年份信息，应该注明年份信息，如：近三年资产负债率（2021年至2023年）：[80.07, 79.23, 82.91]
- 如果需要其它相关上下文中的数据，直接在代码中定义数据，不要把完整的原文本数据写入代码，而是直接将需要的部分数据对应的值写入代码中。
- 必须调用工具`execute_code_tool`来执行代码，并确保执行结果符合问题要求，如果执行错误请根据修改建议修改代码并重新执行
- 不要直接生成代码到你的回答中，而是使用tool call的方式使用工具`execute_code_tool`来执行代码
- 不要虚构任何数据，所有数据都必须是上下文中的数据，如果上下文没有对应的数据，则直接返回`DONE`来结束这次任务
- 代码执行完成后，不要返回代码执行结果或者直接回答用户的问题等内容，直接返回`DONE`来结束这次任务
"""

execute_success_prompt="""
执行你的代码，代码如下：
```python
{imports}
{code}
```
返回结果为：
```bash
{execute_result}
```
- 当你认为代码正确执行，并完成了回答问题所需的计算时，不要返回代码执行结果或者回答用户问题等其它内容，直接返回`DONE`来结束这次任务
"""

execute_error_prompt = """
执行代码：
```python
{imports}
{code}
```
Error-出现错误:
错误信息: 
```error
{execute_error_message}
```
重新生成代码，重新使用工具函数执行
"""

review_code_result = """
错误修改建议：
{review_suggestions}

请根据修改建议修改代码并使用工具执行，修改错误的代码并重新使用工具执行。
"""

over_max_iterations_prompt="""
你已经迭代了{max_iterations}次，直接告诉用户你迭代了{max_iterations}次，并总结现状，简单回复已经完成的指标计算。
"""
