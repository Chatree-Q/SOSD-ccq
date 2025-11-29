import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# --- 核心辅助函数 ---

def bytes_to_unicode():
    """复现GPT-2的bytes_to_unicode映射逻辑"""
    chars = []
    # 1. 收集可打印字符
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = [chr(n) for n in bs]
    
    # 2. 补充剩余字符
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(chr(256 + n))
            n += 1
            
    cs = [chr(n) for n in bs]
    return dict(zip(bs, cs))

def get_stats(ids_list: List[List[int]], counts: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """
    统计当前所有词中的 pair 频率。
    ids_list: [[id, id, ...], [id, id, ...]] 对应每个唯一的词
    counts: {word_idx: freq} 对应每个词的出现频率
    """
    stats = {}
    for idx, ids in enumerate(ids_list):
        freq = counts[idx]
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] = stats.get(pair, 0) + freq
    return stats

# --- Problem 3: BPE 训练函数 (优化版) ---

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 使用标准的 GPT-2 Regex
    # 注意：这里的 Regex 必须与 GPT-2 完全一致，否则分词结果（特别是空格的处理）会有微小差异，导致合并顺序不同
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)
    
    # 2. 预处理：生成词频并转换为初始 ID
    # encoder/decoder 初始状态：0-255 映射到 单字节
    encoder = {bytes([i]): i for i in range(256)}
    decoder = {i: bytes([i]) for i in range(256)}
    
    # 处理特殊 token
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in encoder:
                new_id = len(encoder)
                encoder[token_bytes] = new_id
                decoder[new_id] = token_bytes

    # 统计词频
    # GPT-2 的逻辑是：先按正则切分，然后对切分出的每一块（word）进行 BPE
    chunks = pat.findall(data)
    word_freqs = {}
    for chunk in chunks:
        chunk_bytes = chunk.encode('utf-8')
        word_freqs[chunk_bytes] = word_freqs.get(chunk_bytes, 0) + 1
    
    # 将数据结构转换为更高效的形式进行训练
    # word_ids_list: 存储每个唯一词当前的 token ID 序列
    # word_counts: 存储每个唯一词的频率
    sorted_words = sorted(word_freqs.keys()) # 排序以保证确定性
    word_ids_list = [[b for b in word] for word in sorted_words]
    word_counts = [word_freqs[word] for word in sorted_words]
    
    merges = []
    
    # 3. 主循环 (优化版)
    # 不在每次循环中重新计算所有 stats，而是只计算受影响的部分，但为了代码简洁和通过 1.5s 测试，
    # 针对 500 vocab size，只要不重复对原始文本做正则匹配，直接在 word_ids_list 上操作通常足够快。
    
    current_vocab_size = 256 + (len(special_tokens) if special_tokens else 0)
    
    while current_vocab_size < vocab_size:
        # 计算当前所有 pair 的频率
        stats = get_stats(word_ids_list, word_counts)
        
        if not stats:
            break
            
        # 找到频率最高的 pair
        # 注意：使用 python 的 max 是稳定的，但为了完全确定性，建议按 (freq, pair) 排序
        # 这里的 key 逻辑确保了当频率相同时，按 pair 的字节序/数值序排列
        best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        
        # 记录合并规则
        p1, p2 = best_pair
        # 关键修复：确保 merges 存的是 bytes 对象，而不是 int
        merges.append((decoder[p1], decoder[p2]))
        
        # 生成新 token
        new_id = current_vocab_size
        new_bytes = decoder[p1] + decoder[p2]
        decoder[new_id] = new_bytes
        encoder[new_bytes] = new_id
        
        # 更新 word_ids_list
        # 只需遍历列表，把所有出现的 (p1, p2) 替换为 new_id
        new_word_ids_list = []
        for ids in word_ids_list:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p1 and ids[i+1] == p2:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_word_ids_list.append(new_ids)
        word_ids_list = new_word_ids_list
        
        current_vocab_size += 1
        
    return decoder, merges

# --- Problem 5: Tokenizer 类实现 ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab # id -> bytes
        self.encoder = {v: k for k, v in vocab.items()} # bytes -> id
        
        # 标准 GPT-2 Regex
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_pattern = None
        self.special_encoder = {}
        
        if self.special_tokens:
            self.special_encoder = {t: self.encoder.get(t.encode('utf-8')) for t in special_tokens if t.encode('utf-8') in self.encoder}
            # 过滤掉没在 vocab 里的特殊 token 避免报错（或者按需抛出异常）
            valid_specials = [t for t in self.special_tokens if t in self.special_encoder]
            if valid_specials:
                pattern_str = "|".join(map(regex.escape, sorted(valid_specials, key=len, reverse=True)))
                self.special_pattern = regex.compile(f"({pattern_str})")

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        vocab = {}
        for k, v in vocab_json.items():
            # JSON key 总是 str，需要转 int
            # JSON value 是 unicode 字符串 (用 unicode_escape 编码保存的 bytes)，需要还原为 bytes
            vocab[int(k)] = v.encode('utf-8').decode('unicode_escape').encode('latin1')

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n') # 不要 strip() 导致空格丢失
                if not line: continue
                parts = line.split(' ')
                if len(parts) != 2: continue
                p1, p2 = parts
                # 还原 merge 规则中的 bytes
                p1_bytes = p1.encode('utf-8').decode('unicode_escape').encode('latin1')
                p2_bytes = p2.encode('utf-8').decode('unicode_escape').encode('latin1')
                merges.append((p1_bytes, p2_bytes))
        
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path: str, merges_path: str):
        # 保存时，将 bytes 转为 unicode_escape 字符串，以兼容 JSON
        vocab_json = {k: v.decode('latin1').encode('unicode_escape').decode('utf-8') for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
            
        with open(merges_path, 'w', encoding='utf-8') as f:
            for p1, p2 in self.merges:
                p1_str = p1.decode('latin1').encode('unicode_escape').decode('utf-8')
                p2_str = p2.decode('latin1').encode('unicode_escape').decode('utf-8')
                f.write(f"{p1_str} {p2_str}\n")

    def _bpe_merge(self, word_bytes: bytes) -> List[int]:
        if word_bytes in self.cache:
            return self.cache[word_bytes]
        
        # 将 bytes 转为初始 id 列表
        ids = [self.encoder[bytes([b])] for b in word_bytes]
        
        while len(ids) >= 2:
            stats = {}
            for i in range(len(ids) - 1):
                pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]
            
            if not stats:
                break
            
            best_pair = min(stats, key=stats.get)
            p1_bytes, p2_bytes = best_pair
            merged_bytes = p1_bytes + p2_bytes
            new_id = self.encoder[merged_bytes]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (self.vocab[ids[i]] == p1_bytes) and (self.vocab[ids[i+1]] == p2_bytes):
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        self.cache[word_bytes] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        token_ids = []
        if self.special_pattern:
            # 使用 split 保留分隔符（即特殊 token）
            chunks = self.special_pattern.split(text)
            for i, chunk in enumerate(chunks):
                if i % 2 == 1: # 特殊 token
                    token_ids.append(self.special_encoder[chunk])
                elif chunk: # 普通文本
                    for word in self.pat.findall(chunk):
                        token_ids.extend(self._bpe_merge(word.encode('utf-8')))
        else:
            for word in self.pat.findall(text):
                token_ids.extend(self._bpe_merge(word.encode('utf-8')))
        return token_ids

    def encode_iterable(self, text_iterable: Iterable[str]) -> Iterator[int]:
        for text in text_iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        # 简单拼接 bytes 后解码
        res = b"".join([self.vocab[i] for i in ids])
        return res.decode('utf-8', errors='replace')

if __name__ == '__main__':
    # 简易测试块
    data = "Hello world! This is a test."
    vocab, merges = train_bpe(data, 300)
    tokenizer = BPE_Tokenizer(vocab, merges)
    encoded = tokenizer.encode(data)
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    assert decoded == data
