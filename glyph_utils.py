import matviz
from matviz.etl import write_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import re
from gardiner2unicode import GardinerToUnicodeMap


 # code for translating
def translate_long_text(source_lang, target_lang, text, max_chars=128):
    def translate_chunk(source_lang, target_lang, text):
        with torch.no_grad():
            tokenizer.src_lang = lang_to_m2m_lang_id[source_lang]
            tokenizer.tgt_lang = lang_to_m2m_lang_id[target_lang]
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(lang_to_m2m_lang_id[target_lang]),
                num_beams=10,
                max_length=128
            )
            return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    lang_to_m2m_lang_id = {
        'ea': 'ar', 'tnt': 'ar', 'en': 'en', 'de': 'de', 'lKey': 'my', 'wordClass': 'th'
    }


    words = text.split()
    total_chars = sum(len(word) for word in words) + len(words) - 1  # Total characters including spaces
    num_chunks = -(-total_chars // max_chars)  # Ceiling division to get number of chunks
    chars_per_chunk = total_chars / num_chunks  # Average characters per chunk

    chunks = []
    current_chunk = []
    current_length = 0
    target_length = chars_per_chunk

    for word in words:
        if current_length + len(word) > target_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            target_length += chars_per_chunk
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    translations = [translate_chunk(source_lang, target_lang, chunk) for chunk in chunks]

    # print(f"Chunks: {len(chunks)}, sizes: {[len(c) for c in chunks]}")
    # [print(w) for w in chunks]
    return ' '.join(translations)


def gardiner_to_hieroglyphics(gardiner_string):
    g2u = GardinerToUnicodeMap()

    # Regular expression to find valid Gardiner codes (e.g., "A1", "R4", etc.)
    # This can be adjusted based on the exact valid pattern for Gardiner codes
    gardiner_pattern = re.compile(r'[A-Za-z]+[0-9]+')

    def convert_line(line):
        # Extract all Gardiner codes that match the pattern and convert them
        codes = gardiner_pattern.findall(line)
        return ''.join(chr(int(g2u.to_unicode_hex(code), 16)) for code in codes)

    # Process each line separately, preserving line breaks
    return '\n'.join(convert_line(line) for line in gardiner_string.splitlines())





# Load model directly
tokenizer = AutoTokenizer.from_pretrained("mattiadc/hiero-transformer")
model = AutoModelForSeq2SeqLM.from_pretrained("mattiadc/hiero-transformer")


# # Example usage
# print(gardiner_to_hieroglyphics("A1 A2\nA3 A4"))

# # Example usage
# test_string = 'R4-X8 M23-X1 [&-[&-X8-&]-&]-R4 E15 W18-X1-R8-O21'
# print(gardiner_to_hieroglyphics(test_string))
