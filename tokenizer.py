import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from collections import defaultdict

# --- 核心算法：增量更新 BPE ---

def get_stats(ids_list: List[List[int]], counts: List[int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Dict[int, int]]]:
    """
    初始全量统计。
    返回:
    1. stats: {(p0, p1): total_freq}  ->用于选出最佳 pair
    2. pair_index: {(p0, p1): {word_idx: count_in_word}} -> 倒排索引，记录pair在哪些词里出现
    """
    stats = defaultdict(int)
    pair_index = defaultdict(lambda: defaultdict(int))
    
    for idx, (ids, freq) in enumerate(zip(ids_list, counts)):
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] += freq
            pair_index[pair][idx] += 1
    return stats, pair_index

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 初始化 Vocab
    encoder = {bytes([i]): i for i in range(256)}
    decoder = {i: bytes([i]) for i in range(256)}
    
    special_token_set = set()
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in encoder:
                new_id = len(encoder)
                encoder[token_bytes] = new_id
                decoder[new_id] = token_bytes
                special_token_set.add(token)

    # 2. 预处理文本 (保留特殊 Token 分割逻辑)
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)

    training_data_chunks = []
    if special_token_set:
        escaped_specials = [regex.escape(t) for t in special_tokens]
        special_pat = regex.compile(f"({'|'.join(escaped_specials)})")
        parts = special_pat.split(data)
        for part in parts:
            if part in special_token_set:
                continue 
            if part:
                training_data_chunks.extend(pat.findall(part))
    else:
        training_data_chunks = pat.findall(data)

    # 3. 统计词频
    word_counts_map = defaultdict(int)
    for w in training_data_chunks:
        word_counts_map[w.encode('utf-8')] += 1
            
    # --- 关键修改 1: 排序策略优化 (解决 index 64 logic error) ---
    # 先按频率降序，频率相同时按字典序。这通常是 BPE 的标准 tie-breaking。
    sorted_words = sorted(word_counts_map.keys(), key=lambda x: (word_counts_map[x], x), reverse=True)
    
    word_ids_list = [[encoder[bytes([b])] for b in w_bytes] for w_bytes in sorted_words]
    word_freqs = [word_counts_map[w] for w in sorted_words]
    
    # 4. 构建初始索引 (Solving Speed Issue)
    # stats: 全局对频次
    # pair_index: 倒排索引 {(p0,p1): {word_idx: count}}
    stats, pair_index = get_stats(word_ids_list, word_freqs)
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 5. 增量更新主循环
    while current_vocab_size < vocab_size:
        if not stats:
            break
            
        # 选出频率最高的 pair
        # Python 3.7+ 字典保持插入序，配合初始排序，能保证确定性
        best_pair = max(stats, key=stats.get)
        total_freq = stats[best_pair]
        
        # 记录 Merge
        p0, p1 = best_pair
        merges.append((decoder[p0], decoder[p1]))
        
        # 更新 Vocab
        new_id = current_vocab_size
        new_token_bytes = decoder[p0] + decoder[p1]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # --- 关键修改 2: 增量更新 (Index-based update) ---
        # 只处理包含 best_pair 的那些单词
        indices_to_update = pair_index[best_pair] # {word_idx: count}
        
        # 遍历受影响的每个单词
        for word_idx, _ in indices_to_update.items():
            ids = word_ids_list[word_idx]
            freq = word_freqs[word_idx]
            
            new_ids = []
            i = 0
            while i < len(ids):
                # 找到合并点
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    # --- 1. 扣除旧 Pair 的统计 ---
                    # 此时，旧序列是 ... a, p0, p1, b ...
                    # 合并成 ... a, new_id, b ...
                    # 受影响的旧 Pair 是 (a, p0) 和 (p1, b)
                    # best_pair (p0, p1) 本身会在循环结束后被 delete，这里不用管
                    
                    if i > 0:
                        prev_pair = (ids[i-1], ids[i]) # (a, p0)
                        stats[prev_pair] -= freq
                        pair_index[prev_pair][word_idx] -= 1
                        if pair_index[prev_pair][word_idx] == 0:
                            del pair_index[prev_pair][word_idx]
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]

                    if i < len(ids) - 2:
                        next_pair = (ids[i+1], ids[i+2]) # (p1, b)
                        stats[next_pair] -= freq
                        pair_index[next_pair][word_idx] -= 1
                        if pair_index[next_pair][word_idx] == 0:
                            del pair_index[next_pair][word_idx]
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                    
                    # --- 2. 插入新 Token ---
                    new_ids.append(new_id)
                    
                    # --- 3. 增加新 Pair 的统计 ---
                    # 新序列 ... a, new_id, b ...
                    # 新增 Pair 是 (a, new_id) 和 (new_id, b)
                    
                    if i > 0:
                        new_prev_pair = (ids[i-1], new_id) # (a, new_id)
                        stats[new_prev_pair] += freq
                        pair_index[new_prev_pair][word_idx] += 1
                        
                    if i < len(ids) - 2: # 注意这里还是用原 ids 长度判断后续是否存在
                        # 此时 ids[i+2] 就是 b
                        new_next_pair = (new_id, ids[i+2]) # (new_id, b)
                        stats[new_next_pair] += freq
                        pair_index[new_next_pair][word_idx] += 1
                        
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            # 更新该单词的 IDs
            word_ids_list[word_idx] = new_ids
            
        # 清理已合并的 pair
        del stats[best_pair]
        del pair_index[best_pair]
        
        current_vocab_size += 1
        
    return decoder, merges

# --- Problem 5: Tokenizer 类实现 (保持不变) ---

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
                if i % 2 == 1: 
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
