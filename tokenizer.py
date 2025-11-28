import json
import os
import re
import regex # 使用第三方regex库以更好地支持\p{L}等Unicode属性
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from collections import defaultdict 

# --- 核心辅助函数 ---

def bytes_to_unicode():
    """复现GPT-2的bytes_to_unicode映射逻辑"""
   # 第一步：收集可打印ASCII字符（33-126）和Latin-1可打印字符（161-255）
    chars = []
    for i in range(ord('!'), ord('~') + 1):
        chars.append(chr(i))
    for i in range(ord('¡'), ord('¬') + 1):
        chars.append(chr(i))
    for i in range(ord('®'), ord('ÿ') + 1):
        chars.append(chr(i))
    
    # 第二步：补充剩余字符（用特殊符号填充，确保总长度256）
    n = 0
    while len(chars) < 256:
        if n not in chars:  # 避免重复
            chars.append(chr(n))
        n += 1
    
    # 第三步：生成0-255到chars的映射
    byte_to_char = {i: chars[i] for i in range(256)}
    return byte_to_char

def test_bytes_to_unicode_consistency():
    # 加载参考映射
    with open("bytes_to_unicode_reference.json", "r") as f:
        reference_mapping = json.load(f)
    # 生成自定义映射
    custom_mapping = bytes_to_unicode()
    # 转换为相同格式（如字节值为键，字符为值）
    reference = {int(k): v for k, v in reference_mapping.items()}
    # 逐键对比
    assert custom_mapping == reference, "映射表与参考不一致"



# 全局映射表
byte_to_unicode = bytes_to_unicode()
unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

#def get_pair_stats_optimized(word_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
#    """
#    从词频字典中高效地计算所有相邻字节对的频率。
#    这是优化的关键：我们不遍历整个文本，而是遍历词汇表并乘以其频率。
#    """
#    stats = {}
#    for word, freq in word_freqs.items():
#        for i in range(len(word) - 1):
#            pair = (word[i], word[i+1])
#            stats[pair] = stats.get(pair, 0) + freq
#    return stats
def init_pair_stats(word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    stats = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word)-1):
            stats[(word[i], word[i+1])] += freq
    return stats

def merge_word_freqs_optimized(word_freqs: Dict[Tuple[str, ...], int], pair: Tuple[str, str], new_char: str) -> Dict[Tuple[str, ...], int]:
    new_word_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            if i < n-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_char)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] += freq
    return new_word_freqs


def pretokenize_chunk(text_chunk: str, pat_str: str) -> Dict[Tuple[str, ...], int]:
    """并行化预分词的工作函数（返回Unicode字符tuple的词频）"""
    pat = regex.compile(pat_str)
    word_freqs = defaultdict(int)  # 改为defaultdict(int)
    for word_str in pat.findall(text_chunk):
        word_bytes = word_str.encode("utf-8")
        word_chars = tuple(byte_to_unicode[b] for b in word_bytes)
        word_freqs[word_chars] += 1
    return word_freqs


def update_pair_stats(word: Tuple[str, ...], old_pair: Tuple[str, str], new_char: str, stats: Dict[Tuple[str, str], int], freq: int):
    i = 0
    n = len(word)
    while i < n:
        if i < n-1 and word[i] == old_pair[0] and word[i+1] == old_pair[1]:
            # 移除旧pair
            if stats[old_pair] >= freq:
                stats[old_pair] -= freq
                if stats[old_pair] == 0:
                    del stats[old_pair]
            # 更新左侧新pair
            if i > 0:
                left_pair = (word[i-1], new_char)
                stats[left_pair] += freq
                old_left_pair = (word[i-1], word[i])
                if stats[old_left_pair] >= freq:
                    stats[old_left_pair] -= freq
                    if stats[old_left_pair] == 0:
                        del stats[old_left_pair]
            # 更新右侧新pair
            if i < n-2:
                right_pair = (new_char, word[i+2])
                stats[right_pair] += freq
                old_right_pair = (word[i+1], word[i+2])
                if stats[old_right_pair] >= freq:
                    stats[old_right_pair] -= freq
                    if stats[old_right_pair] == 0:
                        del stats[old_right_pair]
            i += 2
        else:
            i += 1
            

# --- Problem 3: BPE 训练函数 ---



def train_bpe(data, vocab_size, special_tokens):
    # 初始化：文本转字节序列（整数列表，如"new" → [110, 101, 119]）
    tokens = list(data.encode('utf-8'))
    
    # 初始化vocab：特殊token → ID（优先），单字节 → ID
    vocab = {}
    # 1. 添加特殊token
    for idx, token in enumerate(special_tokens):
        vocab[token.encode('utf-8')] = idx  # 特殊token转bytes作为键
    # 2. 添加单字节token（0-255）
    next_token_id = len(special_tokens)
    for byte in range(256):
        vocab[bytes([byte])] = next_token_id
        next_token_id += 1
    
    merges = []  # 最终存储(bytes, bytes)格式的合并规则
    pair_freq = defaultdict(int)
    
    # 初始化pair频率
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        pair_freq[pair] += 1

    while len(vocab) < vocab_size and pair_freq:
        # 1. 找频率最高的pair（整数对）
        best_pair = max(pair_freq, key=pair_freq.get)  # 如(101, 32)
        
        # 2. 将best_pair转为bytes对，加入merges（匹配测试格式）
        p1_bytes = bytes([best_pair[0]])
        p2_bytes = bytes([best_pair[1]])
        merges.append((p1_bytes, p2_bytes))  # 直接存bytes对，无需后续转换
        
        # 3. 新增合并后的token到vocab
        merged_token = p1_bytes + p2_bytes
        new_id = next_token_id  # 新ID
        vocab[merged_token] = new_id
        next_token_id += 1
        
        # 4. 合并token序列（整数列表）
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                # 用合并后的token的ID？不，这里继续用整数表示（或用新ID，看逻辑）
                # 注意：如果后续统计需要，这里可以用新ID，但更简单的是暂时用tuple标记，最后统一转bytes
                new_tokens.append(new_id)  # 或用next_token_id-1（刚分配的ID）
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        
        # 5. 更新pair频率（优化：清空重统计，小数据量足够快）
        pair_freq.clear()
        for i in range(len(tokens)-1):
            # 处理合并后的token（如果是tuple，转为统一的key）
            left = tokens[i] if isinstance(tokens[i], int) else tuple(tokens[i])
            right = tokens[i+1] if isinstance(tokens[i+1], int) else tuple(tokens[i+1])
            pair_freq[(left, right)] += 1

    # 注意：如果测试需要vocab是ID→token，这里反转
    # vocab_id_to_token = {v: k for k, v in vocab.items()}
    return vocab, merges

   


# --- Problem 5: Tokenizer 类实现 (已修正) ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        # 将 merges 转为字典，值为优先级（越小越优先）
        # 键是 (bytes, bytes)
        self.merges = {tuple(pair): i for i, pair in enumerate(merges)} 
        
        # 预编译正则表达式
        PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = regex.compile(PAT_STR)

        # 构建编码器：bytes -> ID
        self.encoder = {v: k for k, v in vocab.items()}
        # 构建解码器：ID -> bytes
        self.decoder = vocab

        # 处理特殊token
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_pattern = None
        self.special_encoder = {}
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(map(re.escape,sorted_special_tokens))
            self.special_pattern = regex.compile(f"({pattern_str})")
            # 建立特殊token的字符串到ID的映射
            for token_str in self.special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes in self.encoder:
                    self.special_encoder[token_str] = self.encoder[token_bytes]

        # 缓存
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            # 关键：加载 JSON 时，Key 是字符串，必须转 int；Value 是字符串，必须转 bytes
            # 注意：这里的 decode('unicode_escape').encode('latin1') 是为了还原被 json 序列化时的字节
            vocab = {}
            for k, v in vocab_json.items():
                vocab[int(k)] = v.encode('utf-8').decode('unicode_escape').encode('latin1')

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # merges文件通常是 "tokenA tokenB"
                parts = line.split()
                if len(parts) != 2: continue # 跳过空行或格式错误的行
                p1, p2 = parts
                p1_bytes = p1.encode('utf-8').decode('unicode_escape').encode('latin1')
                p2_bytes = p2.encode('utf-8').decode('unicode_escape').encode('latin1')
                merges.append((p1_bytes, p2_bytes))
        
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str, merges_filepath: str):
        # 保存词汇表
        vocab_json_save = {k: v.decode('latin1').encode('unicode_escape').decode('utf-8') for k, v in self.vocab.items()}
        with open(vocab_filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_json_save, f, ensure_ascii=False, indent=2)

        # 保存合并规则
        with open(merges_filepath, 'w', encoding='utf-8') as f:
            for p1, p2 in self.merges.keys():
                 p1_str = p1.decode('latin1').encode('unicode_escape').decode('utf-8')
                 p2_str = p2.decode('latin1').encode('unicode_escape').decode('utf-8')
                 f.write(f"{p1_str} {p2_str}\n")
    
    def _bpe_merge(self, word_bytes: bytes) -> List[int]:
        """
        对单个单词进行BPE合并。
        注意：这里操作的是 Token ID，而不是原始字节值。
        """
        if word_bytes in self.cache:
            return self.cache[word_bytes]

        # 1. 初始步骤：将字节序列转换为 ID 序列
        # 修正代码：查 encoder 表
        tokens = [self.encoder[bytes([b])] for b in word_bytes]

        while len(tokens) >= 2:
            # 寻找当前 tokens 列表中所有相邻对中，优先级最高（rank值最小）的一对
            stats = {}
            for i in range(len(tokens) - 1):
                # 获取相邻两个 ID 对应的字节序列
                p1_bytes = self.decoder[tokens[i]]
                p2_bytes = self.decoder[tokens[i+1]]
                pair = (p1_bytes, p2_bytes)
                
                # 如果这个对在合并规则里，记录它的优先级
                if pair in self.merges:
                    stats[pair] = self.merges[pair]

            # 如果没有可合并的对，退出循环
            if not stats:
                break

            # 找到优先级最高（数值最小）的对
            best_pair = min(stats, key=stats.get)
            
            # 计算合并后的新 Token 的 ID
            # 注意：合并后的 bytes = p1_bytes + p2_bytes
            merged_bytes = best_pair[0] + best_pair[1]
            new_id = self.encoder[merged_bytes]

            # 执行合并：在 tokens 列表中替换掉所有的 best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 检查是否是我们要合并的对
                # 需要再次查表确认 current bytes 是否匹配
                if i < len(tokens) - 1:
                    b1 = self.decoder[tokens[i]]
                    b2 = self.decoder[tokens[i+1]]
                    if (b1, b2) == best_pair:
                        new_tokens.append(new_id)
                        i += 2
                        continue
                
                new_tokens.append(tokens[i])
                i += 1
            
            tokens = new_tokens
        
        self.cache[word_bytes] = tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        """将字符串编码为 token ID 列表"""
        token_ids = []
        
        # 处理特殊 token
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
            for i, chunk in enumerate(chunks):
                if i % 2 == 1: # 特殊 token
                    if chunk in self.special_encoder:
                        token_ids.append(self.special_encoder[chunk])
                    else:
                        print(f"Warning: Special token {chunk} not found in vocab.")
                else: # 普通文本
                    if chunk:
                        for word in self.pat.findall(chunk):
                            word_bytes = word.encode("utf-8")
                            token_ids.extend(self._bpe_merge(word_bytes))
        else:
            for word in self.pat.findall(text):
                word_bytes = word.encode("utf-8")
                token_ids.extend(self._bpe_merge(word_bytes))

        return token_ids

    # --- 新增的方法：Problem 6 要求 ---
    def encode_iterable(self, text_iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个文本迭代器进行编码。
        这用于处理大型数据集，避免一次性加载所有文本。
        """
        for text in text_iterable:
            yield from self.encode(text) #返回整数ID

    def decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码为字符串"""
        # 注意：使用 self.decoder 把 ID 转回 bytes
        # errors='replace' 防止非法的 UTF-8 序列导致崩溃
        all_bytes = b"".join(self.decoder[i] for i in ids)
        text = all_bytes.decode("utf-8", errors='replace')
        return text

        
# --- 主执行块 (用于测试和演示) ---
if __name__ == '__main__':
    # 修复 3: 恢复了数据生成代码，避免 FileNotFoundError
    import time
    import resource
    import os

    # 1. 准备训练数据
    INPUT_PATH = "train_dummy.txt" 
    
    # 如果文件不存在，我们就现场造一个！
    if not os.path.exists(INPUT_PATH):
        print(f"正在生成测试数据到 {INPUT_PATH} ...")
        with open(INPUT_PATH, "w", encoding="utf-8") as f:
            data = f.read()  # 读取文件内容
        vocab, merges = train_bpe(data, VOCAB_SIZE, SPECIAL_TOKENS)  # 传入内容而非路径
            f.write("low low low low low\n")
            f.write("lower lower widest widest widest\n")
            f.write("newest newest newest newest newest newest\n")
            f.write("This is a simple test. Emoji: 😊. Chinese: 这里有一些中文测试数据。\n")
            f.write("The quick brown fox jumps over the lazy dog. " * 50)

    # 训练参数
    VOCAB_SIZE = 500
    SPECIAL_TOKENS = ["<|endoftext|>"]
  

    # (a) 训练分词器
    print("开始训练BPE分词器...")
    start_time = time.time()
    
    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 报告训练时间和内存占用
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # in MB
    print(f"\n训练完成！")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"峰值内存占用: {memory_usage:.2f} MB")

    # 词汇表中最长的 token
    longest_token = max(vocab.values(), key=len)
    print(f"词汇表中最长的 token (bytes): {longest_token}")
    print(f"词汇表中最长的 token (str): '{longest_token.decode('utf-8', 'replace')}'")

    # 保存训练结果
    VOCAB_FILE = "tinystories_vocab.json"
    MERGES_FILE = "tinystories_merges.txt"
    tokenizer_for_saving = BPE_Tokenizer(vocab, merges, SPECIAL_TOKENS)
    tokenizer_for_saving.save(VOCAB_FILE, MERGES_FILE)
    print(f"词汇表已保存到 {VOCAB_FILE}")
    print(f"合并规则已保存到 {MERGES_FILE}")
    
    # --- Problem 5 & 6: 使用Tokenizer类 ---
    print("\n--- Tokenizer 实验 ---")
    
    # 关键修改：直接从内存加载，避免保存/读取时的编码问题
    tokenizer = BPE_Tokenizer(vocab, merges, SPECIAL_TOKENS)
    
    # 测试编码和解码
    text_to_test = "newest low lower 😊你好<|endoftext|>"
    encoded = tokenizer.encode(text_to_test)
    decoded = tokenizer.decode(encoded)
    
    print(f"原始文本: '{text_to_test}'")
    print(f"编码结果 (token IDs): {encoded}")
    print(f"解码结果: '{decoded}'")
    
    if text_to_test == decoded:
        print("✅ 编码 -> 解码 一致性测试通过！")
    else:
        print("❌ 警告：解码不匹配")

    # (a) 计算压缩率
    sample_text = "This is a sample document from TinyStories dataset to calculate the compression ratio."
    encoded_sample = tokenizer.encode(sample_text)
    num_bytes = len(sample_text.encode("utf-8"))
    num_tokens = len(encoded_sample)
    compression_ratio = num_bytes / num_tokens
    print(f"\n(a) 压缩率 (bytes/token): {compression_ratio:.2f} ({num_bytes} bytes / {num_tokens} tokens)")

    # (b) 估算吞吐量
    large_text = sample_text * 1000
    start_time_enc = time.time()
    tokenizer.encode(large_text)
    end_time_enc = time.time()
    duration_enc = end_time_enc - start_time_enc
    if duration_enc > 0:
        throughput = len(large_text.encode("utf-8")) / duration_enc / 1e6 # MB/s
        print(f"(b) 编码吞吐量: {throughput:.2f} MB/s")
