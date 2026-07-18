//
//  tokenizer.hpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#ifndef TOKENIZER_hpp
#define TOKENIZER_hpp

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <iostream>
// #include <string_view>
#include <cstdint>
#include <cstring>
class string_view_ {
public:
    string_view_() : data_(nullptr), size_(0) {}
    string_view_(const char* data) : data_(data), size_(static_cast<uint32_t>(std::strlen(data))) {}
    string_view_(const char* data, uint32_t size) : data_(data), size_(size) {}
    string_view_(const std::string& str) : data_(str.data()), size_(static_cast<uint32_t>(str.size())) {}
    constexpr string_view_(const string_view_&) noexcept = default;
    string_view_& operator=(const string_view_&) noexcept = default;
    const char& operator[](uint32_t pos) const { return data_[pos]; }
    constexpr const char* data() const noexcept { return data_; }
    constexpr uint32_t size() const noexcept { return size_; }
    constexpr bool empty() const { return size_ == 0; }
    std::string to_string() const { return std::string(data_, size_); }
    bool operator==(const string_view_& other) const noexcept {
        return size_ == other.size_ && strncmp(data_, other.data_, size_) == 0;
    }
    void remove_prefix(uint32_t n) {
        if (n < size_) {
            data_ += n;
            size_ -= n;
        } else {
            data_ = "";
            size_ = 0;
        }
    }
private:
    const char* data_;
    uint32_t size_ = 0;
};
// std::string_view impl in c++11 end

namespace std {
    template<>
    class hash<string_view_> {
    public:
        size_t operator()(const string_view_& sv) const {
            size_t result = 0;
            for (uint32_t i = 0; i < sv.size(); ++i) {
                result = (result * 31) + static_cast<size_t>(sv[i]);
            }
            return result;
        }
    };
}
namespace MNN {
namespace Transformer {

// std::string_view impl in c++11 start


class Trie {
public:
    struct TrieNode
    {
        std::unordered_map<char, int> children;
        int id = -1;
    };
private:
    // ---- Build-time representation (node + per-node hash map). Released by finalize(). ----
    std::vector<TrieNode> list;
    int size = 1;
    // ---- Read-optimized CSR representation, populated by finalize(). ----
    // Children of node n occupy child_byte_/child_node_[child_offset_[n] .. child_offset_[n+1]),
    // sorted ascending by byte. node_id_[n] is the token id ending at node n (-1 if none).
    std::vector<int> node_id_;
    std::vector<int> child_offset_;
    std::vector<unsigned char> child_byte_;
    std::vector<int> child_node_;
    // Direct 256-way dispatch for the root (node 0), hit at the start of every token match.
    int root_child_[256];
    int getFree() {
        if (static_cast<size_t>(size) >= list.size()) {
            list.resize(list.size()*2);
        }
        return size++;
    }
    void insert(int nid, int token_id, std::string::const_iterator it, std::string::const_iterator end) {
        auto& node = list[nid];
        if (it==end) {
            if (node.id==-1) { node.id=token_id; }
            return;
        }
        auto cid = node.children.find(*it);
        if (cid==node.children.end()) {
            int new_id = getFree();
            list[nid].children.insert({*it, new_id}); // access the node again even after reallocation!!!
            insert(new_id, token_id, it+1, end);
        } else{
            insert(cid->second, token_id, it+1, end);
        }
    }
public:
    Trie(int initial_size=10000) {
        list.resize(initial_size); // init the allocate size
        size = 1; // root
        for (int i = 0; i < 256; ++i) { root_child_[i] = -1; }
    }
    // Pre-grow the node pool so the build never reallocates `list` (each reallocation can copy every
    // per-node hash map). `n` is an upper-bound estimate of the final node count.
    void reserve(int n) {
        if (static_cast<int>(list.size()) < n) { list.resize(n); }
    }
    void insert(std::pair<const std::string&, int> entry) {
        insert(0, entry.second, entry.first.begin(), entry.first.end());
    }
    // Compact the build-time node/hash-map trie into the flat CSR layout used by find(), then free
    // the build-time structures. Must run once after all insert()s and before any find(). Children
    // are sorted by byte so find() can use ordered linear search (small nodes) or binary search.
    void finalize() {
        const int n = size;
        node_id_.resize(n);
        child_offset_.resize(n + 1);
        int total = 0;
        for (int i = 0; i < n; ++i) {
            node_id_[i] = list[i].id;
            child_offset_[i] = total;
            total += static_cast<int>(list[i].children.size());
        }
        child_offset_[n] = total;
        child_byte_.resize(total);
        child_node_.resize(total);
        std::vector<std::pair<unsigned char, int>> tmp;
        for (int i = 0; i < n; ++i) {
            tmp.clear();
            tmp.reserve(list[i].children.size());
            for (const auto& kv : list[i].children) {
                tmp.emplace_back(static_cast<unsigned char>(kv.first), kv.second);
            }
            std::sort(tmp.begin(), tmp.end(),
                      [](const std::pair<unsigned char, int>& a, const std::pair<unsigned char, int>& b) {
                          return a.first < b.first;
                      });
            const int off = child_offset_[i];
            for (int k = 0; k < static_cast<int>(tmp.size()); ++k) {
                child_byte_[off + k] = tmp[k].first;
                child_node_[off + k] = tmp[k].second;
                if (i == 0) { root_child_[tmp[k].first] = tmp[k].second; }
            }
        }
        std::vector<TrieNode>().swap(list); // release build-time memory
    }
    // Greedy longest-match lookup over the finalized CSR trie. Walks input bytes, remembering the
    // deepest node carrying a token id, and on a mismatch rewinds `it` to just past that longest
    // match. Advances `it` by one when the first byte matches nothing (so callers looping on a -1
    // result still progress). Unlike the former version it never dereferences past `end`.
    int find(std::string::const_iterator& it, const std::string::const_iterator& end) {
        if (it == end) { return -1; }
        int nid = 0;
        int current_matched = -1;
        std::string::const_iterator current_it = it + 1;
        for (;;) {
            const int node_id = node_id_[nid];
            if (node_id != -1) {
                current_matched = node_id;
                current_it = it;
            }
            if (it == end) {
                // Input exhausted: cannot descend. Return the longest match found so far.
                if (node_id != -1) { return node_id; }
                it = current_it;
                return current_matched;
            }
            const unsigned char c = static_cast<unsigned char>(*it);
            int child = -1;
            if (nid == 0) {
                child = root_child_[c]; // O(1) root dispatch
            } else {
                int lo = child_offset_[nid];
                int hi = child_offset_[nid + 1];
                if (hi - lo <= 8) {
                    for (int k = lo; k < hi; ++k) {
                        const unsigned char cb = child_byte_[k];
                        if (cb == c) { child = child_node_[k]; break; }
                        if (cb > c) { break; } // children sorted: byte can't appear later
                    }
                } else {
                    while (lo < hi) {
                        const int mid = (lo + hi) >> 1;
                        const unsigned char mb = child_byte_[mid];
                        if (mb < c) { lo = mid + 1; }
                        else if (mb > c) { hi = mid; }
                        else { child = child_node_[mid]; break; }
                    }
                }
            }
            if (child != -1) {
                nid = child;
                ++it;
            } else {
                if (node_id != -1) { return node_id; }
                it = current_it;
                return current_matched;
            }
        }
    }
};


class Tokenizer {
public:
    static constexpr int MAGIC_NUMBER = 430;
    enum TokenizerType {
        SENTENCEPIECE = 0,
        TIKTOIKEN = 1,
        BERT = 2,
        HUGGINGFACE = 3,
        PIPELINE = 4
    };
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    static Tokenizer* createTokenizer(const std::string& filename);
    std::vector<int> encode(const std::string& str);
    std::vector<int> encode(const std::string& str, size_t max_tokens);
    virtual std::string decode(int id) = 0;
    // Zero-copy single-token decode for the streaming hot path: returns a reference to the token's raw
    // bytes when the backend stores them contiguously (Tiktoken overrides this to hand back its decoder
    // table entry with no copy). The default routes through decode(int) using a caller-owned scratch
    // string so backends that synthesize their text keep working unchanged.
    virtual const std::string& decode_id(int id, std::string& scratch) {
        scratch = decode(id);
        return scratch;
    }
protected:
    void cache_special_tokens();
    virtual void load_special(std::ifstream& file);
    virtual bool load_vocab(std::ifstream& file) = 0;
    virtual void encode(const std::string& str, std::vector<int>& ids) = 0;
    virtual bool encode_limited(const std::string& str, std::vector<int>& ids,
                                size_t max_tokens) {
        encode(str, ids);
        if (ids.size() > max_tokens) {
            ids.resize(max_tokens);
            return false;
        }
        return true;
    }
    std::vector<int> special_tokens_;
    std::vector<int> prefix_tokens_;
    std::vector<std::pair<std::string, int>> special_tokens_cache_;
    // First-byte dispatch over special_tokens_cache_: bucket b holds every cached special token whose
    // first byte is b, in cache order. Lets encode() reject, in O(1), any input position whose byte
    // cannot begin a special token instead of rescanning all special tokens at every character.
    std::array<std::vector<std::pair<std::string, int>>, 256> special_first_byte_buckets_;
};

class Sentencepiece : public Tokenizer {
public:
    Sentencepiece() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    enum PieceType {
        NORMAL = 1,
        UNKNOWN = 2,
        CONTROL = 3,
        USER_DEFINED = 4,
        UNUSED = 5,
        BYTE = 6
    };
    struct SentencePiece {
        std::string piece;
        float score;
        PieceType type = PieceType::NORMAL;
        SentencePiece() {}
        SentencePiece(const std::string& p, float s, PieceType t) : piece(p), score(s), type(t) {}
    };
    using EncodeResult = std::vector<std::pair<string_view_, int>>;
private:
    // byte fall back enable
    bool byte_fall_back_ = true;
    // unknown id.
    int unk_id_ = 0;
    // pieces from model
    std::vector<SentencePiece> sentence_pieces_;
    // piece -> id map for normal pieces
    std::unordered_map<string_view_, int> pieces_;
    // piece -> id map for control, unknown, and byte pieces
    std::unordered_map<string_view_, int> reserved_id_map_;
private:
    float get_score(int id) const;
    bool is_unused(int id) const;
    bool is_control(int id) const;
    int piece_to_id(string_view_ w) const;
    std::string byte_to_piece(unsigned char c) const;
    EncodeResult bpe_encode(string_view_ str, float alpha = 0.f);
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual std::string decode(int id) override;
    virtual const std::string& decode_id(int id, std::string& scratch) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    virtual bool encode_limited(const std::string& str, std::vector<int>& ids,
                                size_t max_tokens) override;
    Trie encoder_;
    std::vector<std::string> decoder_;
};

class BertTokenizer : public Tokenizer {
public:
    BertTokenizer() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
    int unk_id_ = -1;   // [UNK] token id resolved from the vocab in load_vocab (-1 = absent)
private:
    std::vector<int> word_piece(const std::string& token);
};

class HuggingfaceTokenizer : public Tokenizer {
struct hash_pair_wstring {
    size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
        auto hash1 = std::hash<std::wstring>{}(p.first);
        auto hash2 = std::hash<std::wstring>{}(p.second);
        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};
using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;
public:
    HuggingfaceTokenizer() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    static void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
    BPERanks bpe_ranks_;
    // Byte-level (GPT-2) alphabet maps each of the 256 byte values to a fixed Unicode code point
    // (all < 0x144). Flat arrays replace the hash maps: O(1) lookups, no hashing, cache-friendly.
    wchar_t b2u_[256] = {};   // byte -> unicode code point
    int16_t u2b_[512];        // unicode code point -> byte (-1 = unmapped); filled in load_vocab
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

struct PreTokenizedString {
    std::vector<std::string> splits;
};

class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string normalize(const std::string& text) const = 0;
};

class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    virtual void pre_tokenize(PreTokenizedString& pts) const = 0;
};

class TokenizerModel {
public:
    virtual ~TokenizerModel() = default;
    virtual std::vector<int> tokenize(const std::string& text) const = 0;
    virtual std::string id_to_token(int id) const = 0;
    virtual size_t vocab_size() const = 0;
};

class PipelineTokenizer : public Tokenizer {
public:
    PipelineTokenizer();
    virtual ~PipelineTokenizer();
    virtual std::string decode(int id) override;
    bool load_vocab_binary(std::ifstream& file);
protected:
    virtual bool load_vocab(std::ifstream&) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    std::unique_ptr<Normalizer> normalizer_;
    std::unique_ptr<PreTokenizer> pre_tokenizer_;
    std::unique_ptr<TokenizerModel> model_;
    struct AddedToken { int id; std::string content; bool special; bool lstrip; bool rstrip; };
    std::vector<AddedToken> added_tokens_;
    std::vector<std::string> added_token_strings_;
    std::string binary_buf_;  // holds binary file data for zero-copy token references
    bool byte_level_ = false; // true if model uses byte-level encoding (GPT-2 style)
    bool wordpiece_decode_ = false; // true if model uses WordPiece decoder
    std::string wordpiece_prefix_; // "##" typically
};

};
};

#endif // TOKENIZER_hpp
