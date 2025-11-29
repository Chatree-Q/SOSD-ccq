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

def get_word_freqs(data: str) -> Dict[bytes, int]:
    """
    预处理文本，返回字节级别的词频统计
    :param data: 输入文本字符串
    :return: {bytes词: 频率}
    """
    # 复用BPE_Tokenizer的正则模式（需确保与分词逻辑一致）
    PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = regex.compile(PAT_STR)
    
    # 1. 按正则切分文本为基础chunk
    chunks = pat.findall(data)
    # 2. 转为字节序列，统计词频
    word_freqs = {}
    for chunk in chunks:
        word_bytes = chunk.encode('utf-8')  # 字符串转字节
        word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    return word_freqs


def get_pair_freq(word_token_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    统计token序列中相邻对的频率
    :param word_token_freqs: {(token_ID序列,): 频率}
    :return: {(token1_ID, token2_ID): 频率}
    """
    pair_freq = {}
    for token_seq, freq in word_token_freqs.items():
        if len(token_seq) < 2:
            continue  # 单个token无相邻对
        # 遍历相邻token对
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i+1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
    return pair_freq



def merge_tokens(word_token_freqs: Dict[Tuple[int, ...], int], 
                 best_pair: Tuple[int, int], 
                 new_id: int) -> Dict[Tuple[int, ...], int]:
    """
    将词序列中的best_pair替换为new_id，返回更新后的词频
    """
    new_word_token_freqs = {}
    for token_seq, freq in word_token_freqs.items():
        new_seq = []
        i = 0
        while i < len(token_seq):
            # 匹配到best_pair则替换为new_id，跳过下一个token
            if i < len(token_seq)-1 and (token_seq[i], token_seq[i+1]) == best_pair:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        # 更新词频（合并相同序列）
        new_seq_tuple = tuple(new_seq)
        new_word_token_freqs[new_seq_tuple] = new_word_token_freqs.get(new_seq_tuple, 0) + freq
    return new_word_token_freqs

            

# --- Problem 3: BPE 训练函数 ---

def train_bpe(data: str, vocab_size: int, special_tokens: Optional[List[str]] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 初始化基础映射：单字节token（0-255）
    encoder: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
    decoder: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    
    # 2. 处理特殊token（添加到映射中）
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in encoder:
                new_id = len(encoder)
                encoder[token_bytes] = new_id
                decoder[new_id] = token_bytes
    
    # 3. 预处理数据：得到词频（word: bytes，freq: int）
    word_freqs = get_word_freqs(data)  # 假设你有这个函数，返回{bytes: int}
    
    # 4. 将词（bytes）转换为token ID序列
    word_token_freqs = {}
    for word, freq in word_freqs.items():
        # 每个字节转对应的ID（依赖encoder）
        token_sequence = [encoder[bytes([b])] for b in word]
        word_token_freqs[tuple(token_sequence)] = freq
    
    # 5. 初始化合并规则和统计
    merges: List[Tuple[bytes, bytes]] = []
    
    # 6. BPE合并循环（直到达到目标词汇量）
    while len(encoder) < vocab_size:
        pair_freq = get_pair_freq(word_token_freqs)
        if not pair_freq:
            break  # 无更多可合并的对
        
        # 找到频率最高的token对
        best_pair = max(pair_freq, key=pair_freq.get)
        p1_id, p2_id = best_pair
        
        # 从decoder中获取ID对应的字节序列
        p1_bytes = decoder[p1_id]
        p2_bytes = decoder[p2_id]
        merged_bytes = p1_bytes + p2_bytes
        
        # 添加新token到映射
        new_id = len(encoder)
        encoder[merged_bytes] = new_id
        decoder[new_id] = merged_bytes
        
        # 更新合并规则
        merges.append((p1_bytes, p2_bytes))
        
        # 更新词的token序列和频率统计
        word_token_freqs = merge_tokens(word_token_freqs, best_pair, new_id)
    
    # 返回ID→字节的vocab和合并规则
    return decoder, merges


   


# --- Problem 5: Tokenizer 类实现 (已修正) ---

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        # 将 merges 转为字典，值为优先级（越小越优先）
        # 键是 (bytes, bytes)
        self.merges = {pair: i for i, pair in enumerate(merges)}  
        
        # 预编译正则表达式
        PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = regex.compile(PAT_STR)
        
        # 构建解码器：ID -> bytes
        self.decoder = vocab
        # 构建编码器：bytes -> ID
        self.encoder = {v: k for k, v in vocab.items()}
      

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
                else:
                    # 这里插入警告/报错逻辑（二选一即可）
                    # 抛出错误（推荐，强制确保特殊token存在）
                    raise ValueError(f"Special token '{token_str}' (bytes: {token_bytes}) not found in vocab!")
  

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

    def save(self, vocab_path: str, merges_path: str):
        # 保存词汇表
        vocab_json_save = {k: v.decode('latin1').encode('unicode_escape').decode('utf-8') for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_json_save, f, ensure_ascii=False, indent=2)
        # 保存合并规则
        merges_list = [list(pair) for pair in self.merges.keys()]
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_list, f, indent=2)
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
            f.write("low low low low low\n")
            f.write("lower lower widest widest widest\n")
            f.write("newest newest newest newest newest newest\n")
            f.write("This is a simple test. Emoji: 😊. Chinese: 这里有一些中文测试数据。\n")
            f.write("The quick brown fox jumps over the lazy dog. " * 50)
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = f.read()  # 读取文件内容
        vocab, merges = train_bpe(data, VOCAB_SIZE, SPECIAL_TOKENS)  # 传入内容而非路径
        
    # 训练参数
    VOCAB_SIZE = 500
    SPECIAL_TOKENS = ["<|endoftext|>"]
  

    # (a) 训练分词器
    print("开始训练BPE分词器...")
    start_time = time.time()
    
    vocab, merges = train_bpe(data, VOCAB_SIZE, SPECIAL_TOKENS)
    
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
