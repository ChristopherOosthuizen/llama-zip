import argparse
import base64
import codecs
import signal
import sys

import numpy as np
import torch
from more_itertools import consume
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PUA_START = 0xE000

NUM_STATE_BITS = 64
FREQ_SCALE_FACTOR = 1 << 32

BASE64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
BASE64_EQ = BASE64 + b"="


class ArithmeticCoderBase:
    def __init__(self):
        full_range = 1 << NUM_STATE_BITS
        self.half_range = full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, cum_freqs, symbol):
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        symhigh = int(cum_freqs[symbol])
        self.high = self.low + symhigh * range // total - 1
        symlow = int(cum_freqs[symbol - 1]) if symbol > 0 else 0
        self.low = self.low + symlow * range // total
        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = (self.low << 1) & self.state_mask
            self.high = ((self.high << 1) & self.state_mask) | 1
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


class Encoder(ArithmeticCoderBase):
    def __init__(self):
        super().__init__()
        self.encoded_data = bytearray()
        self.bit_index = 8
        self.num_underflow = 0

    def get_encoded(self):
        return self.encoded_data

    def encode_symbol(self, cum_freqs, symbol):
        self.update(cum_freqs, symbol)

    def finish(self):
        self.append_bit(1)

    def shift(self):
        bit = self.low >> (NUM_STATE_BITS - 1)
        self.append_bit(bit)
        for _ in range(self.num_underflow):
            self.append_bit(bit ^ 1)
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1

    def append_bit(self, bit):
        if self.bit_index == 8:
            self.encoded_data.append(0)
            self.bit_index = 0
        self.encoded_data[-1] |= bit << (7 - self.bit_index)
        self.bit_index += 1


class Decoder(ArithmeticCoderBase):
    def __init__(self, data: bytes):
        super().__init__()
        self.input = data
        self.byte_index = 0
        self.bit_index = 0
        self.code = sum(
            self.read_code_bit() << i for i in range(NUM_STATE_BITS - 1, -1, -1)
        )

    def decode_symbol(self, cum_freqs):
        total = int(cum_freqs[-1])
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
        symbol = np.searchsorted(cum_freqs, value, side="right")
        self.update(cum_freqs, symbol)
        return symbol

    def shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

    def underflow(self):
        self.code = (
            (self.code & self.half_range)
            | ((self.code << 1) & (self.state_mask >> 1))
            | self.read_code_bit()
        )

    def read_code_bit(self):
        if self.byte_index >= len(self.input):
            return 0
        bit = (self.input[self.byte_index] >> (7 - self.bit_index)) & 1
        self.bit_index = (self.bit_index + 1) % 8
        if self.bit_index == 0:
            self.byte_index += 1
        return bit


# Based on Rust's std::str::Utf8Chunks
class Utf8Chunks:
    def __init__(self, source: bytes):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        if not self.source:
            raise StopIteration

        TAG_CONT_U8 = 128

        def safe_get(xs, i):
            try:
                return xs[i]
            except IndexError:
                return 0

        i = 0
        valid_up_to = 0
        while i < len(self.source):
            byte = self.source[i]
            i += 1

            if byte < 0x80:
                # ASCII
                pass
            elif 0xC2 <= byte <= 0xDF:
                # 2-byte sequence
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xE0 <= byte <= 0xEF:
                # 3-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xE0 == byte and 0xA0 <= next_byte <= 0xBF:
                    pass
                elif 0xE1 <= byte <= 0xEC and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xED == byte and 0x80 <= next_byte <= 0x9F:
                    pass
                elif 0xEE <= byte <= 0xEF and 0x80 <= next_byte <= 0xBF:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            elif 0xF0 <= byte <= 0xF4:
                # 4-byte sequence
                next_byte = safe_get(self.source, i)
                if 0xF0 == byte and 0x90 <= next_byte <= 0xBF:
                    pass
                elif 0xF1 <= byte <= 0xF3 and 0x80 <= next_byte <= 0xBF:
                    pass
                elif 0xF4 == byte and 0x80 <= next_byte <= 0x8F:
                    pass
                else:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
                if safe_get(self.source, i) & 192 != TAG_CONT_U8:
                    break
                i += 1
            else:
                break

            valid_up_to = i

        inspected, remaining = self.source[:i], self.source[i:]
        self.source = remaining

        valid, invalid = inspected[:valid_up_to], inspected[valid_up_to:]
        return Utf8Chunk(valid, invalid)


class Utf8Chunk:
    def __init__(self, valid: bytes, invalid: bytes):
        self.valid = valid
        self.invalid = invalid


def bytes_to_utf8(data: bytes):
    output = bytearray()
    for chunk in Utf8Chunks(data):
        for char in chunk.valid.decode("utf-8"):
            if PUA_START <= ord(char) <= PUA_START + 0xFF:
                for byte in char.encode("utf-8"):
                    output.extend(chr(PUA_START + byte).encode("utf-8"))
            else:
                output.extend(char.encode("utf-8"))
        for byte in chunk.invalid:
            output.extend(chr(PUA_START + byte).encode("utf-8"))
    return bytes(output)


def utf8_to_bytes(data: str):
    output = bytearray()
    for char in data:
        if PUA_START <= ord(char) <= PUA_START + 0xFF:
            output.append(ord(char) - PUA_START)
        else:
            output.extend(char.encode("utf-8"))
    return bytes(output)


class LlamaZip:
    def __init__(
        self, model_path, n_ctx=0, n_gpu_layers=-1, use_mlock=False, verbose=False
    ):
        self.verbose = verbose
        self.load_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
        )

    def load_model(self, model_path, n_ctx, n_gpu_layers, use_mlock):
        loading_message = "Loading model..."
        if self.verbose:
            print(loading_message, end="", flush=True, file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        # Set max context length if specified
        if n_ctx > 0:
            self.model.config.max_position_embeddings = n_ctx
        if self.verbose:
            print(
                "\r" + " " * len(loading_message) + "\r",
                end="",
                flush=True,
                file=sys.stderr,
            )

    def compute_cdf(self, logits):
        # logits: torch.Tensor of shape (1, vocab_size)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs).detach().cpu().numpy()[0]
        freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
        cum_freqs = np.cumsum(freqs)
        return cum_freqs

    def tokenizer_adds_space_prefix(self):
        space = " "
        double_space = "  "
        tokenized = self.tokenizer.encode(space, add_special_tokens=False)
        detokenized = self.tokenizer.decode(
            tokenized, clean_up_tokenization_spaces=False
        )
        return detokenized == double_space

    def compress(self, uncompressed: bytes, window_overlap=0) -> bytes:
        uncompressed_text = bytes_to_utf8(uncompressed).decode("utf-8")
        tokens = self.tokenizer.encode(uncompressed_text, add_special_tokens=False)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.sep_token_id
        if eos_token_id is None:
            raise ValueError("No EOS token found in tokenizer.")
        tokens.append(eos_token_id)

        token_encoder = Encoder()

        progress_bar = tqdm(
            total=len(tokens),
            mininterval=1 / 30,
            desc="Compressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
            disable=not self.verbose,
        )

        interrupted = False

        def sigint_handler(*_):
            nonlocal interrupted
            interrupted = True

        s = signal.signal(signal.SIGINT, sigint_handler)

        next_token_idx = 0
        max_context_length = self.model.config.max_position_embeddings

        while next_token_idx < len(tokens):
            start_idx = max(0, next_token_idx - window_overlap)
            context_tokens = tokens[start_idx:next_token_idx]
            input_ids = [self.tokenizer.bos_token_id] + context_tokens
            input_ids = torch.tensor([input_ids])

            if input_ids.size(1) >= max_context_length:
                # Truncate input_ids if necessary
                input_ids = input_ids[:, -max_context_length:]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]  # get logits for the next token

            # Compute cdf
            logits = logits.unsqueeze(0)
            cdf = self.compute_cdf(logits)

            # Get the next token
            if interrupted and next_token_idx < len(tokens) - 1:
                next_token_idx = len(tokens) - 1
                if self.verbose:
                    print(file=sys.stderr)
            next_token = tokens[next_token_idx]
            next_token_idx += 1

            # Encode the next token
            token_encoder.encode_symbol(cdf, next_token)

            progress_bar.update()

        progress_bar.close()
        token_encoder.finish()
        compressed = token_encoder.get_encoded()
        signal.signal(signal.SIGINT, s)
        return compressed

    def decompress(self, compressed: bytes, window_overlap=0) -> bytes:
        seen_tokens = []
        decompressed = bytearray()
        token_decoder = Decoder(compressed)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.sep_token_id
        if eos_token_id is None:
            raise ValueError("No EOS token found in tokenizer.")

        done = False
        max_context_length = self.model.config.max_position_embeddings

        while not done:
            start_idx = max(0, len(seen_tokens) - window_overlap)
            context_tokens = seen_tokens[start_idx:]
            input_ids = [self.tokenizer.bos_token_id] + context_tokens
            input_ids = torch.tensor([input_ids])

            if input_ids.size(1) >= max_context_length:
                # Truncate input_ids if necessary
                input_ids = input_ids[:, -max_context_length:]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]  # get logits for the next token

            # Compute cdf
            logits = logits.unsqueeze(0)
            cdf = self.compute_cdf(logits)

            # Decode the next token
            next_token = token_decoder.decode_symbol(cdf)

            if next_token == eos_token_id:
                done = True
                break

            # Append the token to seen_tokens
            seen_tokens.append(next_token)

            # Decode the token to utf-8
            next_utf8 = self.tokenizer.decode(
                [next_token], clean_up_tokenization_spaces=False
            )
            # Handle tokenizer adds space prefix
            if (
                len(seen_tokens) == 1
                and next_utf8.startswith(" ")
                and self.tokenizer_adds_space_prefix()
            ):
                next_utf8 = next_utf8[1:]

            # Convert utf-8 to bytes directly
            next_bytes = utf8_to_bytes(next_utf8)
            decompressed.extend(next_bytes)
            if self.verbose:
                sys.stdout.buffer.write(next_bytes)
                sys.stdout.buffer.flush()

        return decompressed


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="LLM-powered lossless compression tool"
    )
    parser.add_argument("model_path", help="path to model file")
    parser.add_argument(
        "-f",
        "--compressed-format",
        choices=["binary", "base64"],
        help="format of compressed data (default: binary, except for interactive mode, which only supports base64)",
    )
    parser.add_argument(
        "-w",
        "--window-overlap",
        dest="overlap",
        default="0%",
        help="how much model context (as number of tokens or percentage of model context length) to maintain after filling the window. higher values increase compression ratio but decrease speed. must use same value for compression and decompression (default: 0%%)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        help="model context length (default: 0, which uses maximum supported by the model)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="number of model layers to offload to GPU (default: -1, which offloads all layers)",
    )
    parser.add_argument(
        "--use-mlock",
        default=False,
        action="store_true",
        help="use mlock to keep model in RAM (disabled by default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose output during compression/decompression",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-c",
        "--compress",
        dest="string",
        nargs="*",
        help="compress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-d",
        "--decompress",
        dest="compressed",
        nargs="*",
        help="decompress argument string (or stdin if no argument is provided)",
    )
    mode_group.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        default=False,
        action="store_true",
        help="show a prompt for interactive compression and decompression",
    )
    return parser


def robust_b64decode(input_bytes):
    filtered_base64 = bytes(byte for byte in input_bytes if byte in BASE64)
    padded_base64 = filtered_base64 + b"A" * (-len(filtered_base64) % 4)
    return base64.b64decode(padded_base64)


def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    if args.compressed_format is None:
        args.compressed_format = "base64" if args.interactive else "binary"
    elif args.interactive and args.compressed_format != "base64":
        parser.error("interactive mode only supports base64 compressed data")

    compressor = LlamaZip(
        model_path=args.model_path,
        use_mlock=args.use_mlock,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
    )

    try:
        if args.overlap.endswith("%"):
            percent = float(args.overlap[:-1])
            if not (0 <= percent <= 100):
                parser.error("window overlap must be in the range [0%, 100%]")
            window_overlap = int(
                percent / 100 * (compressor.model.config.max_position_embeddings - 1)
            )
        else:
            window_overlap = int(args.overlap)
            if window_overlap < 0:
                window_overlap += compressor.model.config.max_position_embeddings
            if not (
                0 <= window_overlap < compressor.model.config.max_position_embeddings
            ):
                parser.error(
                    f"window overlap must be in the range [{-compressor.model.config.max_position_embeddings}, {compressor.model.config.max_position_embeddings - 1}]"
                )
    except ValueError:
        parser.error(
            "window overlap must be an integer (number of tokens) or a percentage (of the model's context length)"
        )

    try:
        if args.string is not None:
            uncompressed = (
                " ".join(args.string).encode("utf-8")
                if args.string
                else sys.stdin.buffer.read()
            )
            compressed = compressor.compress(uncompressed, window_overlap)
            if args.compressed_format == "base64":
                compressed = base64.b64encode(compressed)
            sys.stdout.buffer.write(compressed)
            sys.stdout.buffer.flush()
        elif args.compressed is not None:
            compressed = (
                args.compressed[0].encode("utf-8")
                if args.compressed
                else sys.stdin.buffer.read()
            )
            if args.compressed_format == "base64":
                compressed = robust_b64decode(compressed)
            decompressed = compressor.decompress(compressed, window_overlap)
            if not compressor.verbose:
                sys.stdout.buffer.write(decompressed)
                sys.stdout.buffer.flush()
        elif args.interactive:
            while True:
                try:
                    input_bytes = input("≥≥≥ ").encode("utf-8")
                except UnicodeDecodeError:
                    print(
                        "error: interactive mode only supports UTF-8 input",
                        file=sys.stderr,
                    )
                    continue
                if input_bytes and all(byte in BASE64_EQ for byte in input_bytes):
                    try:
                        compressed = robust_b64decode(input_bytes)
                        decompressed = compressor.decompress(
                            compressed, window_overlap
                        )
                    except KeyboardInterrupt:
                        pass
                    if not compressor.verbose:
                        sys.stdout.buffer.write(decompressed)
                        sys.stdout.buffer.flush()
                else:
                    compressed = compressor.compress(input_bytes, window_overlap)
                    compressed = base64.b64encode(compressed)
                    sys.stdout.buffer.write(compressed)
                    sys.stdout.buffer.flush()
                print("\n", file=sys.stderr)
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()

