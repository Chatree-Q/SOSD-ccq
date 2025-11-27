import json
import os
import re
import regex # 使用第三方regex库以更好地支持\p{L}等Unicode属性
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# --- 核心辅助函数 ---

def get_pair_stats_optimized(word_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    从词频字典中高效地计算所有相邻字节对的频率。
    这是优化的关键：我们不遍历整个文本，而是遍历词汇表并乘以其频率。
    """
    stats = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] = stats.get(pair, 0) + freq
    return stats

def merge_word_freqs(word_freqs: Dict[Tuple[int, ...], int], pair: Tuple[int, int], new_id: int) -> Dict[Tuple[int, ...], int]:
    """
    在词频字典的所有词中，执行一次合并操作。
    这也是优化的关键：我们不修改一个巨大的列表，而是创建一个新的词频字典。
    """
    new_word_freqs = {}
    p1, p2 = pair
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs

def pretokenize_chunk(text_chunk: str, pat_str: str) -> Dict[Tuple[int, ...], int]:
    """并行化预分词的工作函数"""
    pat = regex.compile(pat_str)
    word_freqs = {}
    # 使用 regex.findall 而不是 re.findall
    for word_str in pat.findall(text_chunk):
        word_bytes = tuple(word_str.encode("utf-8"))
        word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    return word_freqs


# --- Problem 3: BPE 训练函数 ---

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练一个字节级的BPE分词器。

    Args:
        input_path: 训练数据路径。
        vocab_size: 目标词汇表大小。
        special_tokens: 特殊token列表。

    Returns:
        A tuple containing:
            - vocab: 从token ID到其字节序列的映射。
            - merges: 按创建顺序列出的BPE合并规则。
    """
    assert vocab_size >= 256 + len(special_tokens), "词汇表大小不足"

    # 1. 词汇表初始化 (Vocabulary initialization)
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        # 将特殊token放在词汇表的末尾，ID从vocab_size-1开始递减
        # 这样可以确保它们不会与合并的token ID冲突
        token_id = vocab_size - 1 - i
        vocab[token_id] = token_str.encode("utf-8")

    # GPT-2的正则表达式
    PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 读取语料库
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 预分词 (Pre-tokenization)
    # 首先按特殊token分割语料库
    special_pattern = "|".join(map(re.escape, special_tokens))
    text_chunks = re.split(f"({special_pattern})", text)

    # 并行化预分词
    num_procs = min(cpu_count(), os.cpu_count() or 1)
    with Pool(num_procs) as pool:
        # 我们只对非特殊token的块进行预分词
        # 奇数索引的块是特殊token本身，偶数索引是普通文本
        work_chunks = [chunk for i, chunk in enumerate(text_chunks) if i % 2 == 0 and chunk]
        
        # 使用 partial 来传递固定的 pat_str 参数（或者像下面这样用lambda/wrapper）
        # from functools import partial
        worker = partial(pretokenize_chunk, pat_str=PAT_STR)
        
        chunk_freqs_list = list(tqdm(
            pool.imap(worker, work_chunks),
            total=len(work_chunks),
            desc="并行预分词"
        ))
    
    # 合并所有进程的结果
    word_freqs = {}
    for chunk_freqs in chunk_freqs_list:
        for word, freq in chunk_freqs.items():
            word_freqs[word] = word_freqs.get(word, 0) + freq

    # 3. 计算 BPE 合并 (Compute BPE merges)
    num_merges = vocab_size - len(vocab)
    merges_list = []  # 使用列表来保证顺序
    
    # 词汇表的ID从256开始增长
    next_token_id = 256
    
    pbar = tqdm(range(num_merges), desc="BPE 合并")
    for i in pbar:
        # (a) 统计所有相邻 token 对的频率
        stats = get_pair_stats_optimized(word_freqs)
        if not stats:
            print("没有更多的对可以合并，提前停止。")
            break

        # (b) 找到频率最高的 token 对，并处理平局
        def tie_break_key(item):
            pair, freq = item
            # 返回一个元组 (频率, 字节对)，Python会按顺序比较
            return (freq, pair)
            
        best_pair = max(stats.items(), key=tie_break_key)[0]

        # (c) 用一个新的 token "AB" 替换所有 ("A", "B") 对
        word_freqs = merge_word_freqs(word_freqs, best_pair, next_token_id)

        # (d) 将 "AB" 添加到词汇表中
        p1_bytes = vocab[best_pair[0]]
        p2_bytes = vocab[best_pair[1]]
        vocab[next_token_id] = p1_bytes + p2_bytes

        # (e) 将 ("A", "B") 记录到合并规则列表 merges 中
        merges_list.append((p1_bytes, p2_bytes))
        pbar.set_description(f"合并 {p1_bytes.decode('utf-8', 'replace')}{p2_bytes.decode('utf-8', 'replace')} -> {next_token_id}")

        next_token_id += 1

    return vocab, merges_list


# --- Problem 5: Tokenizer 类实现 ---
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
        # 你的旧代码：tokens = list(word_bytes)  <-- 错误：这假设了 ID == ByteValue
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
                        # 如果特殊token不在表中，当普通文本处理(很少发生)
                        # 或者在这里报错
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
    # --- Problem 4: 在TinyStories上训练 ---
    import time
    import resource

    # 创建一个虚拟的训练文件
    # dummy_data_path = "TinyStoriesV2-GPT4-valid.txt"
    # with open(dummy_data_path, "w", encoding="utf-8") as f:
    #     f.write("low low low low low\n")
    #     f.write("lower lower widest widest widest\n")
    #     f.write("newest newest newest newest newest newest\n")
    #     f.write("This is a simple test for the BPE tokenizer. It should handle Unicode like 😊 and CJK characters like 你好世界。\n")
    INPUT_PATH = "train1.txt"
    # 训练参数
    VOCAB_SIZE = 5000
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
    tokenizer_for_saving = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    tokenizer_for_saving.save(VOCAB_FILE, MERGES_FILE)
    print(f"词汇表已保存到 {VOCAB_FILE}")
    print(f"合并规则已保存到 {MERGES_FILE}")

    # (b) 性能分析提示:
    print("\n(b) 性能分析提示:")
    print("要对代码进行性能分析，可以使用cProfile模块。")
    print("例如: python -m cProfile -o profile.stats your_script_name.py")
    print("然后使用 `snakeviz profile.stats` 来可视化结果。")
    print("预计 `get_pair_stats_optimized` 和 `merge_word_freqs` 会是合并步骤中的热点。")
    
    # --- Problem 5 & 6: 使用Tokenizer类 ---
    print("\n--- Tokenizer 实验 ---")
    
    # 从文件加载分词器
    tokenizer = Tokenizer.from_files(VOCAB_FILE, MERGES_FILE, SPECIAL_TOKENS)
    
    # 测试编码和解码
    text_to_test = "newest low lower 😊你好<|endoftext|>"
    encoded = tokenizer.encode(text_to_test)
    decoded = tokenizer.decode(encoded)
    
    print(f"原始文本: '{text_to_test}'")
    print(f"编码结果 (token IDs): {encoded}")
    print(f"解码结果: '{decoded}'")
    assert text_to_test == decoded
    print("编码 -> 解码 一致性测试通过！")

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
    throughput = len(large_text.encode("utf-8")) / duration_enc / 1e6 # MB/s
    print(f"(b) 编码吞吐量: {throughput:.2f} MB/s")

    # (c) 为什么 uint16 是合适的数据类型？
    print("\n(c) 为什么 uint16 是合适的数据类型？")
    print(f"一个无符号16位整数 (uint16) 可以表示 2^16 = 65,536 个不同的值 (从 0 到 65,535)。")
    print(f"对于 5K, 10K, 或 32K 大小的词汇表，这个范围完全足够覆盖所有的token ID。")
    print("相比使用默认的int32或int64，使用uint16可以节省一半或更多的内存/磁盘空间，这在处理数十亿级别的token时非常重要。")
