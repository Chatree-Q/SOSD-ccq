import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# --- 核心辅助函数 ---

def get_stats(ids_list: List[List[int]], counts: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """
    统计 Pair 频率，不跨越单词边界。
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

# --- Problem 3: BPE 训练函数 ---

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # --- 调试打印 (如果没看到这句话，说明没运行到这里) ---
    print("\n\n>>> 正在运行新版代码: train_bpe 被调用 <<<") 
    print(f">>> 目标词表大小: {vocab_size}")
    
    # 1. GPT-2 标准正则
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)
    
    # 2. 初始化 Vocab (0-255)
    encoder = {bytes([i]): i for i in range(256)}
    decoder = {i: bytes([i]) for i in range(256)}
    
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in encoder:
                new_id = len(encoder)
                encoder[token_bytes] = new_id
                decoder[new_id] = token_bytes

    # 3. 预处理：生成 Unique Words 列表 (速度优化的关键)
    print(">>> 正在统计词频...")
    words_strings = pat.findall(data)
    word_counts_map = {}
    for w in words_strings:
        w_bytes = w.encode('utf-8')
        word_counts_map[w_bytes] = word_counts_map.get(w_bytes, 0) + 1
    
    sorted_unique_words = sorted(word_counts_map.keys())
    word_ids_list = [[encoder[bytes([b])] for b in word] for word in sorted_unique_words]
    word_freqs = [word_counts_map[word] for word in sorted_unique_words]
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 4. 主循环
    print(">>> 开始 BPE 合并循环...")
    while current_vocab_size < vocab_size:
        stats = get_stats(word_ids_list, word_freqs)
        if not stats:
            break
            
        # Tie-breaking: 频率高优先 -> byte值小优先
        best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        
        p1, p2 = best_pair
        
        # --- 核心修正：必须存入 bytes ---
        # 以前可能存的是 (p1, p2) 即 (int, int)
        # 现在强制转为 bytes
        merges.append((decoder[p1], decoder[p2]))
        
        # 更新 Vocab
        new_id = current_vocab_size
        new_token_bytes = decoder[p1] + decoder[p2]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # 更新 IDs
        word_ids_list = merge_vocab(best_pair, new_id, word_ids_list)
        current_vocab_size += 1
        
    print(f">>> 训练完成，Merges 数量: {len(merges)}")
    return decoder, merges

# --- Problem 5: Tokenizer 类 ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab 
        self.encoder = {v: k for k, v in vocab.items()}
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_pattern = None
        self.special_encoder = {}
        
        if self.special_tokens:
            for t in special_tokens:
                 t_bytes = t.encode('utf-8')
                 if t_bytes in self.encoder:
                     self.special_encoder[t] = self.encoder[t_bytes]
            
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
            vocab[int(k)] = v.encode('utf-8').decode('unicode_escape').encode('latin1')

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line: continue
                parts = line.split(' ')
                if len(parts) != 2: continue
                p1, p2 = parts
                p1_bytes = p1.encode('utf-8').decode('unicode_escape').encode('latin1')
                p2_bytes = p2.encode('utf-8').decode('unicode_escape').encode('latin1')
                merges.append((p1_bytes, p2_bytes))
        
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path: str, merges_path: str):
        vocab_json = {k: v.decode('latin1').encode('unicode_escape').decode('utf-8') for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
            
        with open(merges_path, 'w', encoding='utf-8') as f:
            for p1, p2 in self.merges:
                p1_str = p1.decode('latin1').encode('unicode_escape').decode('utf-8')
                p2_str = p2.decode('latin1').encode('unicode_escape').decode('utf-8')
                f.write(f"{p1_str} {p2_str}\n")

    def _bpe_merge(self, word_bytes: bytes) -> List[int]:
        if word_bytes in self.cache: return self.cache[word_bytes]
        ids = [self.encoder[bytes([b])] for b in word_bytes]
        while len(ids) >= 2:
            stats = {}
            for i in range(len(ids) - 1):
                pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                if pair in self.merges: stats[pair] = self.merges[pair]
            if not stats: break
            best_pair = min(stats, key=stats.get)
            p1, p2 = best_pair
            new_id = self.encoder[p1 + p2]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and self.vocab[ids[i]] == p1 and self.vocab[ids[i+1]] == p2:
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
                if i % 2 == 1: token_ids.append(self.special_encoder[chunk])
                elif chunk:
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
