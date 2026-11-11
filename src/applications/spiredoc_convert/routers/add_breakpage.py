from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re
from loguru import logger


def get_paragraph_text(p_element) -> str:
    parts = []
    for node in p_element.iter():
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
        elif node.tag == qn("w:br"):
            parts.append("\n")
    return "".join(parts)


def make_page_break_paragraph():
    p = OxmlElement("w:p")
    r = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br)
    p.append(r)
    return p


def read_docx_in_order(filepath: str):
    """
    遍历 docx 片段，按标签和阈值规则插入分页符：

    规则1: <lq_N> / <cq_N> 开标签 -> 所在段前加分页符（最外层）
           </lq_N> / </cq_N> 闭标签 -> 所在段后加分页符（最外层）
    规则2: lq/cq 嵌套时，只对最外层标签按规则1执行
    规则3: lq/cq 外部的段落累计达80个 -> 在下一个 </q_N> 后插入分页符
    规则4: 表格累计达5个 -> 在表格后插入分页符

    任意分页符插入后，所有计数重置为0，q_stack 清空。
    """
    try:
        doc = Document(filepath)
        body = doc.element.body
        children = list(body)

        TAG_PATTERN = re.compile(r"<(/?)(lq_|cq_|q_)(\d+)>")

        # 段落阈值
        PARA_THRESHOLD = 50
        # 表格阈值
        TABLE_THRESHOLD = 3
        # 合理化阈值
        FIX_THRESHOLD = 3

        lq_cq_stack = []
        q_stack = []
        q_open_count = 0
        table_count = 0

        actions = []

        for idx, child in enumerate(children):
            tag = child.tag

            if tag == qn("w:p"):
                text = get_paragraph_text(child)

                # 检测段落是否本身是分页符
                # page_breaks = child.findall('.//' + qn('w:br'))
                # if any(br.get(qn('w:type')) == 'page' for br in page_breaks):
                #     actions.append((idx, "init"))

                # if not text.strip():
                #     continue

                tags_found = TAG_PATTERN.findall(text)
                has_lq_cq = any(p in ("lq_", "cq_") for _, p, _ in tags_found)

                # 第一遍：处理 lq/cq 栈 + q 栈
                for closing_slash, prefix, number in tags_found:
                    if prefix in ("lq_", "cq_"):
                        if not closing_slash:
                            lq_cq_stack.append(prefix + number)
                            if len(lq_cq_stack) == 1:
                                actions.append((idx, "before"))
                                q_open_count = 0
                                table_count = 0
                                q_stack.clear()
                                logger.info(f"  [规则1] <{prefix}{number}> 最外层开始 -> 段前分页")
                        else:
                            if lq_cq_stack:
                                lq_cq_stack.pop()
                                if len(lq_cq_stack) == 0:
                                    actions.append((idx, "after"))
                                    q_open_count = 0
                                    table_count = 0
                                    q_stack.clear()
                                    logger.info(f"  [规则1] </{prefix}{number}> 最外层结束 -> 段后分页")

                    elif prefix == "q_":
                        if not closing_slash:
                            q_stack = [prefix + number]
                        else:
                            if q_stack:
                                q_stack.pop()

                # 段落计数：lq/cq 外部的段落才计入
                if not has_lq_cq and len(lq_cq_stack) == 0:
                    q_open_count += 1

                # 第二遍：q 闭标签触发分页（保证 q 标签对不被拆开）
                for closing_slash, prefix, number in tags_found:
                    if prefix == "q_" and closing_slash and q_open_count >= PARA_THRESHOLD:
                        actions.append((idx, "after"))
                        q_open_count = 0
                        table_count = 0
                        q_stack.clear()
                        logger.info(f"  [规则3] </q_{number}> 段后分页")

            elif tag == qn("w:tbl") and len(lq_cq_stack) == 0:
                table_count += 1
                rows = child.findall(qn("w:tr"))
                row_count = len(rows)
                logger.info(f"  [表格 {table_count}] 共 {row_count} 行")
                q_open_count += row_count
                if table_count >= TABLE_THRESHOLD and len(q_stack) == 0:
                    actions.append((idx, "after"))
                    q_open_count = 0
                    table_count = 0
                    q_stack.clear()
                    logger.info(f"  [规则4] 第 {TABLE_THRESHOLD} 个表格后插入分页符")

        # 倒序遍历，避免删除元素导致索引错乱
        logger.info(f"正在处理分页符，共 {len(actions)} 个， 初始 actions: {actions}")

        i = len(actions) - 1

        # 倒序遍历，i>=1 保证 i-1 有效
        while i >= 1:
            prev_idx, prev_pos = actions[i - 1]
            curr_idx, curr_pos = actions[i]

            # 你的原始规则
            if prev_pos == 'after' and curr_pos == 'before' and (curr_idx - prev_idx) <= FIX_THRESHOLD:
                del actions[i - 1]  # 删除前一个 after
            else:
                # 不满足条件，正常往前遍历
                i -= 1

        logger.info(f"处理后分页符，共 {len(actions)} 个，处理后的 actions: {actions}")

        for child_idx, position in reversed(actions):
            child = children[child_idx]
            pb = make_page_break_paragraph()
            if position == "before":
                child.addprevious(pb)
            else:
                child.addnext(pb)

        logger.info(f"\n共插入 {len(actions)} 个分页符")

        output_path = filepath.replace(".docx", "_paged.docx")
        doc.save(output_path)
        logger.info(f"分页符已补充，文件路径：{output_path}")
        return output_path
    except Exception as e:
        logger.error(f"处理分页符失败：{e}")
        return filepath