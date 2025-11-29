import json
import os
import regex
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from collections import defaultdict

# --- 核心算法：安全高效的增量 BPE ---

def get_pair_counts(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """计算单个单词内所有 Pair 的频率"""
    counts = defaultdict(int)
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i+1])] += 1
    return counts

def get_stats(ids_list: List[List[int]], counts: List[int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Dict[int, int]]]:
    """初始全量统计"""
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
            t_bytes = token.encode('utf-8')
            if t_bytes not in encoder:
                nid = len(encoder)
                encoder[t_bytes] = nid
                decoder[nid] = t_bytes
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
            if part in special_token_set: continue 
            if part: training_data_chunks.extend(pat.findall(part))
    else:
        training_data_chunks = pat.findall(data)

    # 3. 统计词频
    word_counts_map = defaultdict(int)
    for w in training_data_chunks:
        word_counts_map[w.encode('utf-8')] += 1
            
    # 排序：虽然下面的 max 逻辑是主要的 tie-breaker，但基础顺序能保证稳定性
    # 使用 (-freq, x) 排序：优先处理高频词
    sorted_words = sorted(word_counts_map.keys(), key=lambda x: (-word_counts_map[x], x))
    
    word_ids_list = [[encoder[bytes([b])] for b in w_bytes] for w_bytes in sorted_words]
    word_freqs = [word_counts_map[w] for w in sorted_words]
    
    # 4. 构建初始索引
    stats, pair_index = get_stats(word_ids_list, word_freqs)
    
    merges = []
    current_vocab_size = len(decoder)
    
    # 5. 主循环
    while current_vocab_size < vocab_size:
        # 清理
        garbage = [k for k, v in stats.items() if v <= 0]
        for k in garbage: del stats[k]

        if not stats: break
            
        # --- 关键策略: Tie-breaking ---
        # 参考实现倾向于 max((freq, pair))，即频率高优先，频率相同字节序大优先
        # 这通过了 index 92 的 (t, h) > (c, om) 测试
        best_pair = max(stats, key=lambda p: (stats[p], p))
        
        p0, p1 = best_pair
        merges.append((decoder[p0], decoder[p1]))
        
        new_id = current_vocab_size
        new_token_bytes = decoder[p0] + decoder[p1]
        decoder[new_id] = new_token_bytes
        encoder[new_token_bytes] = new_id
        
        # --- 鲁棒的增量更新 ---
        indices_to_update = pair_index[best_pair]
        # Copy keys to avoid runtime error if dict changes size
        indices_list = list(indices_to_update.keys())
        
        for word_idx in indices_list:
            ids = word_ids_list[word_idx]
            freq = word_freqs[word_idx]
            
            # 1. 撤销旧统计：直接减去该单词目前贡献的所有 Pair
            # 这样做虽然比精细操作慢一点点，但绝对正确，不会有边界 bug
            if len(ids) >= 2:
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    stats[pair] -= freq
                    pair_index[pair][word_idx] -= 1
                    if pair_index[pair][word_idx] == 0: del pair_index[pair][word_idx]

            # 2. 生成新序列
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            word_ids_list[word_idx] = new_ids
            
            # 3. 应用新统计：加上新序列的所有 Pair
            if len(new_ids) >= 2:
                for i in range(len(new_ids) - 1):
                    pair = (new_ids[i], new_ids[i+1])
                    stats[pair] += freq
                    pair_index[pair][word_idx] += 1
        
        # 确保 best_pair 被清除 (虽然上面的循环应该已经把它减为0了)
        if best_pair in stats: del stats[best_pair]
        
        current_vocab_size += 1
        
    return decoder, merges

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
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
                 if t_bytes in self.encoder: self.special_encoder[t] = self.encoder[t_bytes]
            valid_specials = [t for t in self.special_tokens if t in self.special_encoder]
            if valid_specials:
                pattern_str = "|".join(map(regex.escape, sorted(valid_specials, key=len, reverse=True)))
                self.special_pattern = regex.compile(f"({pattern_str})")
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode('utf-8').decode('unicode_escape').encode('latin1') for k, v in vocab_json.items()}
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split(' ')
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8').decode('unicode_escape').encode('latin1'), parts[1].encode('utf-8').decode('unicode_escape').encode('latin1')))
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path, merges_path):
        vocab_json = {k: v.decode('latin1').encode('unicode_escape').decode('utf-8') for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f: json.dump(vocab_json, f, ensure_ascii=False, indent=2)
        with open(merges_path, 'w', encoding='utf-8') as f:
            for p1, p2 in self.merges:
                p1s = p1.decode('latin1').encode('unicode_escape').decode('utf-8')
                p2s = p2.decode('latin1').encode('unicode_escape').decode('utf-8')
                f.write(f"{p1s} {p2s}\n")

    def _bpe_merge(self, word_bytes):
        if word_bytes in self.cache: return self.cache[word_bytes]
        ids = [self.encoder[bytes([b])] for b in word_bytes]
        while len(ids) >= 2:
            stats = {}
            for i in range(len(ids) - 1):
                pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                if pair in self.merges: stats[pair] = self.merges[pair]
            if not stats: break
            # encode 使用 min rank，即最先合并的优先
            best_pair = min(stats, key=stats.get)
            p1, p2 = best_pair
            new_id = self.encoder[p1 + p2]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and self.vocab[ids[i]] == p1 and self.vocab[ids[i+1]] == p2:
                    new_ids.append(new_id); i += 2
                else: new_ids.append(ids[i]); i += 1
            ids = new_ids
        self.cache[word_bytes] = ids
        return ids

    def encode(self, text):
        token_ids = []
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
            for i, chunk in enumerate(chunks):
                if i % 2 == 1:
                    if chunk in self.special_encoder: token_ids.append(self.special_encoder[chunk])
                elif chunk:
                    for word in self.pat.findall(chunk): token_ids.extend(self._bpe_merge(word.encode('utf-8')))
        else:
            for word in self.pat.findall(text): token_ids.extend(self._bpe_merge(word.encode('utf-8')))
        return token_ids
        
    def encode_iterable(self, text_iterable):
        for text in text_iterable: yield from self.encode(text)

    def decode(self, ids):
        return b"".join([self.vocab[i] for i in ids]).decode('utf-8', errors='replace')
