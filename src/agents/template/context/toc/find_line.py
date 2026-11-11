SERIAL_PATTERNS = [
    r'^\d+$', r'^\d+[\.。,，]\s*', r'^\d+、', r'^\(\d+\)\s*', r'^（\d+）\s*',
    r'^[\u2460-\u2473\u2776-\u277F\u2488-\u249B][、.]?\s*', r'^((\d+\.)+\d+\.?\s*)+',
    r'^[一二三四五六七八九十]+、\s*', r'^（[一二三四五六七八九十]+）\s*',
    r'^\([一二三四五六七八九十]+\)\s*', r'^[IVXLCDM]+\.\s*',
    r'^[A-Za-z]\.\s*', r'^\(\s*[A-Za-z]\s*\)\s*', r'^\([Ⅰ-Ⅻ]+\)\s*',
]

def match_array_to_lines_keywords(char_array:list,text_content:str):
    out = []
    for c in char_array:
        lines = []
        for i,line in enumerate(text_content.split('\n')):
            if c in line:
                lines.append(i)
        out.append((c,lines))
        if len(lines) == 1:
            for i,o in enumerate(out):
                if len(o[1])>1:
                    out[i] = (o[0],[t for t in o[1] if t < lines[0]])
    out = [(o[0],o[1][0]) for o in out if o[1]]
    # print(out)
    return out
