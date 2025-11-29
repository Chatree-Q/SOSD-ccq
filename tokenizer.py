import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# --- 核心辅助函数 ---

def get_stats(ids_list: List[List[int]], counts: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """
    根据当前的词表(ids_list)和词频(counts)统计所有相邻字符对的频率。
    注意：这里不跨越单词边界统计，只在每个单词内部统计。
    """
    stats = {}
    for idx, ids in enumerate(ids_list):
        freq = counts[idx]
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] = stats.get(pair, 0) + freq
    return stats

def merge_vocab(pair: Tuple[int, int], new_id: int, ids_list: List[List[int]]) -> List[List[int]]:
    """
    将 ids_list 中所有的 pair 替换为 new_id
    """
    p0, p1 = pair
    new_ids_list = []
    for ids in ids_list:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        new_ids_list.append(new_ids)
    return new_ids_list

# --- Problem 3: BPE 训练函数 (完全重构) ---

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 严格使用 GPT-2 的正则表达式
    # 你的原代码把 [sdmt] 写在了一起，这会导致分词行为与 GPT-2 不一致，从而导致 merge 顺序错误
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)
    
    # 2. 初始化 Vocab (0-255)
    # 内部逻辑使用 int 进行计算，最后返回时确保转为 bytes
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

    # 3. 预处理：按正则切分并统计词频
    # 这一步是速度优化的关键：我们将几百万字符的 data 压缩为几千个 unique words
    words_strings = pat.findall(data)
    word_counts_map = {}
    for w in words_strings:
        w_bytes = w.encode('utf-8')
        word_counts_map[w_bytes] = word_counts_map.get(w_bytes, 0) + 1
    
    # 转换为 ID 列表以便进行 BPE 训练
    # 排序以确保确定性 (Deterministic)
    sorted_unique_words = sorted(word_counts_map.keys())
    
    # word_ids_list: List[List[int]] -> 每个词对应的 token ID 序列
    word_ids_list = [[encoder[bytes([b])] for b in word] for word in sorted_unique_words]
    # word_freqs: List[int] -> 每个词出现的次数
    word_freqs = [word_counts_map[word] for word in sorted_unique_words]
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 4. 主循环：在 Unique Words 列表上操作，而不是在原始文本上操作
    while current_vocab_size < vocab_size:
        # 统计 pair 频率
        stats = get_stats(word_ids_list, word_freqs)
        
        if not stats:
            break
            
        # 找到频率最高的 pair
        # 使用 tie-breaking 规则：频率高优先；频率相同，ID 小的优先
        best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        
        # 记录 Merge 规则
        p1, p2 = best_pair
        
        # --- 关键修复：确保 merges 存入的是 bytes 对象 ---
        # 你的报错显示存入的是 (int, int)，这里强制查表转换为 bytes
        merges.append((decoder[p1], decoder[p2]))
        
        # 更新词表
        new_id = current_vocab_size
        new_token_bytes = decoder[p1] + decoder[p2]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # 更新所有词的 ID 序列
        word_ids_list = merge_vocab(best_pair, new_id, word_ids_list)
        
        current_vocab_size += 1
        
    return decoder, merges

# --- Problem 5: Tokenizer 类实现 ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab # id -> bytes
        self.encoder = {v: k for k, v in vocab.items()} # bytes -> id
        
        # 必须与 train_bpe 保持一致的 Regex
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_pattern = None
        self.special_encoder = {}
        
        if self.special_tokens:
            self.special_encoder = {}
            for t in special_tokens:
                 # 确保特殊 token 在 vocab 中
                 if t.encode('utf-8') in self.encoder:
                     self.special_encoder[t] = self.encoder[t.encode('utf-8')]
            
            if self.special_encoder:
                pattern_str = "|".join(map(regex.escape, sorted(self.special_encoder.keys(), key=len, reverse=True)))
                self.special_pattern = regex.compile(f"({pattern_str})")

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        vocab = {}
        for k, v in vocab_json.items():
            # 处理 JSON 加载时的编码问题 (unicode_escape -> latin1 -> bytes)
            vocab[int(k)] = v.encode('utf-8').decode('unicode_escape').encode('latin1')

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
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
        # 保存时使用 unicode_escape 确保 bytes 可逆
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
        res = b"".join([self.vocab[i] for i in ids])
        return res.decode('utf-8', errors='replace')

if __name__ == '__main__':
    # 简单的本地测试，确保基本流程没问题
    print("Running basic test...")
    text = "Hello world! This is a test."
    vocab, merges = train_bpe(text, 260) # 训练少量 token
    tokenizer = BPE_Tokenizer(vocab, merges)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {text}")
    print(f"Decoded:  {decoded}")
    assert text == decoded
    print("Basic test passed.")
