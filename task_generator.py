import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_task_from_chunk(text: str) -> dict | None:
    lines = [l.strip() for l in text.split('\n')]

    exercise_pattern = re.compile(r'^(\d+\.\d+)\s+(.+)')
    hint_line_pattern = re.compile(r'^\d+\s+\(([^)]+/[^)]+)\)\s*$')
    numbered_hint_pattern = re.compile(r'^(\d+)\s+\(([^)]+)\)\s*$')

    skip_keywords = [
        "tick", "choose from these verbs", "which sentence",
        "use your own", "write sentences", "match", "look at",
        "what's happening", "picture", "write", "describe"
    ]

    current_exercise = None
    current_instruction = None
    sentences = []
    collecting = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        ex_match = exercise_pattern.match(line)
        if ex_match:
            if sentences and current_exercise:
                break

            ex_text = ex_match.group(2).strip()
            if any(kw in ex_text.lower() for kw in skip_keywords):
                collecting = False
                current_exercise = None
                current_instruction = None
                sentences = []
                i += 1
                continue

            current_exercise = ex_match.group(1)
            current_instruction = ex_text
            sentences = []
            collecting = True
            i += 1
            continue

        if not collecting:
            i += 1
            continue

        num_hint = numbered_hint_pattern.match(line)
        if num_hint:
            num = num_hint.group(1)
            hint_content = num_hint.group(2).strip()

            starter = ""
            j = i + 1
            while j < len(lines) and j < i + 3:
                next_line = lines[j].strip()
                if next_line and not re.match(r'^\d+\s+\(', next_line) and not exercise_pattern.match(next_line):
                    if len(next_line.split()) <= 4 and next_line[0].isupper():
                        starter = next_line
                        break
                elif next_line:
                    break
                j += 1

            if starter:
                sentences.append(f"{num}. {starter} _______ ({hint_content})")
            else:
                sentences.append(f"{num}. _______ ({hint_content})")

            i += 1
            continue

        if re.match(r'^\d+[.)]\s+\(.+/.+\)', line):
            num_match = re.match(r'^(\d+)[.)]\s+\((.+)\)', line)
            if num_match:
                num = num_match.group(1)
                hint_content = num_match.group(2).strip()

                starter = ""
                j = i + 1
                while j < len(lines) and j < i + 3:
                    next_line = lines[j].strip()
                    if next_line and not re.match(r'^\d+\s+\(', next_line) and not exercise_pattern.match(next_line):
                        if len(next_line.split()) <= 4 and next_line[0].isupper():
                            starter = next_line
                            break
                    elif next_line:
                        break
                    j += 1

                if starter:
                    sentences.append(f"{num}. {starter} _______ ({hint_content})")
                else:
                    sentences.append(f"{num}. _______ ({hint_content})")

        i += 1

    if sentences and current_exercise:
        return {
            "exercise": current_exercise,
            "instruction": current_instruction or "Complete the sentences.",
            "sentences": sentences[:8],
        }

    return None


def extract_and_format_task(chunk_text: str, book: str, page: str, llm=None) -> str | None:
    task = extract_task_from_chunk(chunk_text)
    if not task:
        if llm:
            prompt = ChatPromptTemplate.from_template(
                "You are an English teacher. The following text is raw OCR output from a grammar exercise textbook.\n"
                "Clean it up and format it into a simple fill-in-the-blank or multiple-choice exercise.\n"
                "CRITICAL Rules:\n"
                "- Extract ONLY ONE coherent exercise part if there are multiple.\n"
                "- IMPORTANT: If the textbook provides a list of words or options (a 'WORD BANK') at the top or bottom, you MUST include it.\n"
                "- REMOVE any instructions or sentences that require looking at pictures (e.g. 'Look at the pictures'), listening to audio, pointing, or partner work.\n"
                "- Only pick parts that can be solved purely by text.\n"
                "- Format output nicely with '**Exercise**', the instruction in bold, 'Word bank' (if applicable), and nicely numbered sentences.\n"
                "- Do NOT output any answers. Leave blanks as ______.\n"
                "- If no text-only exercise can be formed, reply exactly with 'NO_TASK_FOUND'.\n\n"
                "Raw text:\n{text}\n\nFormatted Exercise:"
            )
            try:
                chain = prompt | llm | StrOutputParser()
                output = chain.invoke({"text": chunk_text})
                if not output.strip() or "NO_TASK_FOUND" in output or output == "NO_TASK":
                    pass 
                else:
                    return f"{output}\n\n*Source: {book}, p.{page}*"
            except Exception:
                pass
                
        lines = [l.strip() for l in chunk_text.split('\n') if l.strip()]
        if len(lines) < 2:
            return "NO_TASK_FOUND" 
            
        output = "**Exercise**\n\n"
        output += "\n".join(lines)
        output += f"\n\n*Source: {book}, p.{page}*"
        return output

    output = f"**Exercise {task['exercise']}**\n\n"
    output += f"**{task['instruction']}**\n\n"
    output += "\n".join(task['sentences'])
    output += f"\n\n*Source: {book}, p.{page}*"
    return output


def parse_generated_task(raw_answer: str, book: str, page: str) -> str:
    if "NO_TASK_FOUND" in raw_answer:
        return "NO_TASK_FOUND"

    try:
        unit_match = re.search(r'UNIT:\s*(\S+)', raw_answer, re.IGNORECASE)
        exercise_match = re.search(r'EXERCISE:\s*(\S+)', raw_answer, re.IGNORECASE)
        instruction_match = re.search(r'INSTRUCTION:\s*(.*?)(?=\nBANK:|\nSENTENCES:|\nRULE_CONTEXT:|\Z)', raw_answer, re.DOTALL | re.IGNORECASE)
        bank_match = re.search(r'BANK:\s*(.*?)(?=\nSENTENCES:|\nRULE_CONTEXT:|\Z)', raw_answer, re.DOTALL | re.IGNORECASE)
        sentences_match = re.search(r'SENTENCES:\s*(.*?)(?=\n[A-Z_]+:|\Z)', raw_answer, re.DOTALL | re.IGNORECASE)

        unit = unit_match.group(1).strip() if unit_match else "?"
        exercise = exercise_match.group(1).strip() if exercise_match else "?"
        instruction = instruction_match.group(1).strip() if instruction_match else "Complete the task"
        bank_str = bank_match.group(1).strip() if bank_match else "NONE"
        sentences_raw = sentences_match.group(1).strip() if sentences_match else ""

        if not sentences_raw:
            return "NO_TASK_FOUND"

        sentence_lines = re.findall(r'^\d+[.)]\s*.+', sentences_raw, re.MULTILINE)
        if not sentence_lines:
            return "NO_TASK_FOUND"

        words_to_hide = []
        if bank_str.upper() != "NONE" and bank_str.strip():
            words_to_hide = [w.strip() for w in bank_str.split(',') if w.strip()]

        blanked_lines = []
        for line in sentence_lines:
            blanked = line
            for w in words_to_hide:
                if len(w) > 1:
                    pattern = re.compile(rf'\b{re.escape(w)}\b', re.IGNORECASE)
                    blanked = pattern.sub('_______', blanked)
            blanked_lines.append(blanked)

        output = f"**Unit {unit} · Exercise {exercise}**\n\n"
        output += f"**{instruction}**\n\n"
        if words_to_hide:
            output += f"**Word bank:** {bank_str}\n\n"
        output += "\n".join(blanked_lines)
        output += f"\n\n*Source: {book}, p.{page}*"
        return output

    except Exception:
        return raw_answer