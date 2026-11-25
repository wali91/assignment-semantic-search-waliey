# Semantic Search + LLM untuk FAQ E-commerce

Prototype asisten AI customer service berbasis knowledge base FAQ, sebagai persiapan menuju RAG system penuh.

## 1. Problem Statement

**Objective.** Tim customer service e-commerce menerima ratusan pertanyaan repetitif setiap hari (pemesanan, pengiriman, pembayaran, pengembalian barang). Prototype ini bertujuan:

1. mencari entri FAQ yang paling relevan terhadap pertanyaan pelanggan, **meskipun kata-katanya tidak persis sama** dengan FAQ (semantic search), dan
2. menyusun jawaban yang natural dengan LLM berdasarkan FAQ yang ditemukan.

**Knowledge base.** `data/ecommerce_faq_knowledge_base.csv` (dibagikan oleh penyelenggara, tidak diubah). Struktur diperiksa di notebook (jumlah entri, kolom, kategori).

> TODO: isi jumlah entri, daftar kolom, dan distribusi kategori setelah notebook dijalankan.

**Batasan dan asumsi.**
- Sistem hanya menjawab berdasarkan FAQ yang tersedia; jika tidak ada, sistem harus menyatakan informasinya belum tersedia.
- Tidak ada riwayat percakapan (setiap query berdiri sendiri).
- Retrieval memakai top-k = 3 tanpa reranking.
- Data hanya dibaca, tidak pernah ditulis ulang.

## 2. Alur Sistem

```
ingest -> embed -> store -> retrieve -> generate

CSV (Pandas) -> unit teks (kategori + pertanyaan + jawaban)
             -> Gemini Embeddings (RETRIEVAL_DOCUMENT) -> vektor
             -> FAISS IndexFlatIP (vektor dinormalisasi L2 = cosine similarity)

Query -> Gemini Embeddings (RETRIEVAL_QUERY) -> FAISS top-k
      -> prompt (Persona, Task, Context, Format) -> Gemini LLM -> jawaban
```

| Tahap | Implementasi |
|---|---|
| Ingest | `pandas.read_csv`, cek struktur, gabungkan kategori + question + answer |
| Embed | `gemini-embedding-001`, 768 dimensi, verifikasi shape |
| Store | `faiss.IndexFlatIP`, verifikasi `index.ntotal == jumlah entri` |
| Retrieve | `semantic_search(query, top_k)` menampilkan rank, score, kategori, pertanyaan |
| Generate | `build_prompt()` + `generate_answer()` (Gemini), digabung di `rag_answer()` |

Prompt memiliki empat bagian: **Persona** (asisten CS ramah), **Task** (jawab hanya dari Context, akui jika tidak ada), **Context** (hasil retrieval), **Format** (Bahasa Indonesia, singkat).

## 3. Hasil Pengujian

> TODO: salin hasil dari bagian "Pengujian" di notebook setelah dijalankan dengan API key Anda.

| Tipe | Query | Top-1 FAQ | Score | Retrieval akurat? | Kualitas jawaban LLM |
|---|---|---|---|---|---|
| Ada di KB | | | | | |
| Ada di KB | | | | | |
| Parafrase | | | | | |
| Parafrase | | | | | |
| Tidak ada di KB | | | | | |

**Analisis:** (query mana yang akurat, mana yang kurang relevan, dan bagaimana LLM merespons query yang tidak ada jawabannya.)

## 4. Keterbatasan

Isi berdasarkan temuan Anda. Contoh yang umum:
- Tidak ada threshold skor, sehingga query di luar KB tetap mengembalikan top-k yang tidak relevan; pencegahan halusinasi sepenuhnya bergantung pada prompt.
- `IndexFlatIP` melakukan pencarian brute-force; kurang efisien untuk KB yang sangat besar.
- Tanpa reranking dan tanpa memori percakapan.
- Kualitas jawaban bergantung pada kelengkapan FAQ.

## 5. Cara Menjalankan

```bash
cd semantic-search-assignment
python -m venv .venv && source .venv/bin/activate   # opsional
pip install -r requirements.txt

export GEMINI_API_KEY="..."       # Windows PowerShell: $env:GEMINI_API_KEY="..."
# letakkan CSV di data/ecommerce_faq_knowledge_base.csv

jupyter notebook notebook/starter_notebook_semantic_search.ipynb
```

Jalankan semua cell dari atas ke bawah. Jika nama model atau kolom CSV berbeda dari starter notebook, sesuaikan `EMBED_MODEL`, `LLM_MODEL`, `Q_COL`, dan `A_COL`.

## Struktur Project

```
semantic-search-assignment/
├── data/ecommerce_faq_knowledge_base.csv
├── notebook/starter_notebook_semantic_search.ipynb
├── documentation/retrieval_test_results.md (opsional)
├── README.md
└── requirements.txt
```
