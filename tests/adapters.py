import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from collections import defaultdict

# --- 核心算法：增量更新 BPE (带 First Seen 追踪) ---

def get_stats(ids_list: List[List[int]], counts: List[int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Dict[int, int]], Dict[Tuple[int, int], int]]:
    """
    初始全量统计。
    返回:
    1. stats: {(p0, p1): total_freq}
    2. pair_index: {(p0, p1): {word_idx: count_in_word}}
    3. first_seen: {(p0, p1): min_word_idx}  -> 记录 Pair 第一次出现的单词索引
    """
    stats = defaultdict(int)
    pair_index = defaultdict(lambda: defaultdict(int))
    first_seen = {}
    
    for idx, (ids, freq) in enumerate(zip(ids_list, counts)):
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] += freq
            pair_index[pair][idx] += 1
            # 记录该 pair 第一次出现的单词索引 (idx 是按顺序递增的)
            if pair not in first_seen:
                first_seen[pair] = idx
                
    return stats, pair_index, first_seen

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

    # 2. 预处理文本
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

    # 3. 统计词频 (使用普通 dict 保持 First Appearance Order)
    word_counts_map = {}
    for w in training_data_chunks:
        w_bytes = w.encode('utf-8')
        if w_bytes in word_counts_map:
            word_counts_map[w_bytes] += 1
        else:
            word_counts_map[w_bytes] = 1
            
    # 不要 sort！保持 word_counts_map 的插入顺序
    # 这样 ids_list[0] 就是文本中第一个遇到的 unique word
    word_ids_list = [[encoder[bytes([b])] for b in w_bytes] for w_bytes in word_counts_map.keys()]
    word_freqs = list(word_counts_map.values())
    
    # 4. 构建初始索引
    stats, pair_index, first_seen = get_stats(word_ids_list, word_freqs)
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 5. 增量更新主循环
    while current_vocab_size < vocab_size:
        if not stats:
            break
            
        # --- 关键修改：First Appearance Tie-breaking ---
        # 1. 频率最高 (-stats[p] 最小)
        # 2. 最早出现 (first_seen[p] 最小)
        best_pair = min(stats, key=lambda p: (-stats[p], first_seen[p]))
        
        # 记录 Merge
        p0, p1 = best_pair
        merges.append((decoder[p0], decoder[p1]))
        
        # 更新 Vocab
        new_id = current_vocab_size
        new_token_bytes = decoder[p0] + decoder[p1]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # 增量更新
        indices_to_update = pair_index[best_pair]
        # 必须排序，确保我们按单词出现顺序处理，从而正确维护新 pair 的 first_seen
        sorted_indices = sorted(indices_to_update.keys())
        
        for word_idx in sorted_indices:
            ids = word_ids_list[word_idx]
            freq = word_freqs[word_idx]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    # 扣除旧 Pair 统计
                    if i > 0:
                        prev_pair = (ids[i-1], ids[i])
                        stats[prev_pair] -= freq
                        pair_index[prev_pair][word_idx] -= 1
                        if pair_index[prev_pair][word_idx] == 0:
                            del pair_index[prev_pair][word_idx]
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                            del first_seen[prev_pair] # 可选，清理内存

                    if i < len(ids) - 2:
                        next_pair = (ids[i+1], ids[i+2])
                        stats[next_pair] -= freq
                        pair_index[next_pair][word_idx] -= 1
                        if pair_index[next_pair][word_idx] == 0:
                            del pair_index[next_pair][word_idx]
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                            del first_seen[next_pair]
                    
                    # 插入新 Token
                    new_ids.append(new_id)
                    
                    # 增加新 Pair 统计
                    if i > 0:
                        new_prev_pair = (ids[i-1], new_id)
                        stats[new_prev_pair] += freq
                        pair_index[new_prev_pair][word_idx] += 1
                        # 维护 first_seen: 因为 word_idx 是从小到大遍历的，
                        # 如果该 pair 还没见过，当前 word_idx 一定是它第一次出现的地方
                        if new_prev_pair not in first_seen:
                            first_seen[new_prev_pair] = word_idx
                        
                    if i < len(ids) - 2:
                        new_next_pair = (new_id, ids[i+2])
                        stats[new_next_pair] += freq
                        pair_index[new_next_pair][word_idx] += 1
                        if new_next_pair not in first_seen:
                            first_seen[new_next_pair] = word_idx
                        
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            word_ids_list[word_idx] = new_ids
            
        del stats[best_pair]
        del pair_index[best_pair]
        del first_seen[best_pair]
        
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
