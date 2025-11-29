import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# --- 核心算法优化 ---

def get_stats(ids_list: List[List[int]], counts: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """
    统计 Pair 频率。
    优化点：使用最原始的字典操作以提升速度。
    """
    stats = {}
    for idx, ids in enumerate(ids_list):
        freq = counts[idx]
        length = len(ids)
        for i in range(length - 1):
            pair = (ids[i], ids[i+1])
            if pair in stats:
                stats[pair] += freq
            else:
                stats[pair] = freq
    return stats

def merge_vocab(pair: Tuple[int, int], new_id: int, ids_list: List[List[int]]) -> List[List[int]]:
    """
    执行合并。
    优化点：仅在包含 p0 的列表上进行操作（虽然 Python 列表遍历已经很快了，这里保持逻辑简单）。
    """
    p0, p1 = pair
    new_ids_list = []
    for ids in ids_list:
        new_ids = []
        i = 0
        length = len(ids)
        while i < length:
            if i < length - 1 and ids[i] == p0 and ids[i+1] == p1:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        new_ids_list.append(new_ids)
    return new_ids_list

# --- Problem 3: BPE 训练函数 ---

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 初始化 Vocab
    # 使用 0-255 初始化
    encoder = {bytes([i]): i for i in range(256)}
    decoder = {i: bytes([i]) for i in range(256)}
    
    # 将特殊 Token 加入 Vocab，但我们需要记住它们以便稍后处理
    special_token_set = set()
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in encoder:
                new_id = len(encoder)
                encoder[token_bytes] = new_id
                decoder[new_id] = token_bytes
                special_token_set.add(token)

    # 2. 预处理文本 (关键修改：处理特殊 Token)
    # 标准 GPT-2 Regex
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)

    # 如果有特殊 token，我们需要确保它们不被 regex 切碎。
    # 策略：先按特殊 token split，只对非特殊 token 部分应用 pat.findall
    # 这样特殊 token 就完全不会进入 BPE 的统计环节（它们已经是完整的 token 了）
    
    training_data_chunks = []
    if special_token_set:
        # 创建一个用于分割特殊 token 的正则
        escaped_specials = [regex.escape(t) for t in special_tokens]
        special_pat = regex.compile(f"({'|'.join(escaped_specials)})")
        # 分割
        parts = special_pat.split(data)
        for part in parts:
            if part in special_token_set:
                continue # 跳过特殊 token，不让它参与频次统计
            if part:
                training_data_chunks.extend(pat.findall(part))
    else:
        # 没有特殊 token，直接切
        training_data_chunks = pat.findall(data)

    # 3. 统计词频 (Unique Words Optimization)
    # 关键修改：不要排序！保持出现顺序，这决定了 Tie-breaking 的行为。
    word_counts_map = {}
    for w in training_data_chunks:
        w_bytes = w.encode('utf-8')
        if w_bytes in word_counts_map:
            word_counts_map[w_bytes] += 1
        else:
            word_counts_map[w_bytes] = 1
            
    # 转为 ID 列表
    # word_ids_list: List[List[int]]
    # word_freqs: List[int]
    # 这里我们利用 Python 3.7+ 字典保持插入顺序的特性
    word_ids_list = []
    word_freqs = []
    for w_bytes, freq in word_counts_map.items():
        word_ids_list.append([encoder[bytes([b])] for b in w_bytes])
        word_freqs.append(freq)
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 4. BPE 主循环
    while current_vocab_size < vocab_size:
        stats = get_stats(word_ids_list, word_freqs)
        
        if not stats:
            break
            
        # 关键修改：只使用 stats.get 作为 key
        # 这意味着如果频率相同，Python 的 max 会返回先遍历到的那个。
        # 由于我们移除了 sorted()，现在的顺序是基于文本中单词出现的顺序，这通常能对齐参考实现。
        best_pair = max(stats, key=stats.get)
        
        p1, p2 = best_pair
        
        # 记录 Merge (必须是 bytes)
        merges.append((decoder[p1], decoder[p2]))
        
        # 更新 Vocab
        new_id = current_vocab_size
        new_token_bytes = decoder[p1] + decoder[p2]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # 更新 IDs
        word_ids_list = merge_vocab(best_pair, new_id, word_ids_list)
        
        current_vocab_size += 1
        
    return decoder, merges

# --- Problem 5: Tokenizer 类实现 ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab 
        self.encoder = {v: k for k, v in vocab.items()}
        # 正则必须一致
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
                # 编译特殊 token 正则用于 encode 时的切分
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
            # encode 时取 min priority，数值越小越优先
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
                if i % 2 == 1: # 是特殊 token
                    if chunk in self.special_encoder:
                         token_ids.append(self.special_encoder[chunk])
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
