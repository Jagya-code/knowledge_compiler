-- ============================================================
-- schema.sql — Knowledge Store Database Schema
-- Run this once to set up the database before loading data.
-- ============================================================

-- Enable pgvector extension for storing/querying embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Drop tables if they exist (clean slate on re-run)
-- ============================================================
DROP TABLE IF EXISTS node_embeddings CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;

-- ============================================================
-- Table 1: nodes
-- Stores every node from the pruned knowledge tree.
-- ============================================================
CREATE TABLE nodes (
    id               SERIAL PRIMARY KEY,
    node_id          VARCHAR(20)   NOT NULL,          -- e.g. "0001"
    title            VARCHAR(500)  NOT NULL,           -- node title
    depth            INT           NOT NULL,           -- 0 = root
    parent_id        INT           REFERENCES nodes(id) ON DELETE CASCADE,  -- NULL for root nodes
    summary          TEXT,                             -- LLM-generated summary
    keywords         TEXT[],                           -- array of keywords
    source_docs      TEXT[],                           -- array of source filenames
    user_defined     BOOLEAN       DEFAULT FALSE,      -- TRUE if matched from input_config
    guidance_context JSONB         DEFAULT '{}',       -- guidance descriptions
    path_string      TEXT          NOT NULL            -- e.g. "Operations > Claims Management > Claims Processing"
);

-- Index for fast prefix search on path_string (used by path-based retrieval)
CREATE INDEX idx_nodes_path_string  ON nodes USING btree (path_string);

-- Index for depth-based queries
CREATE INDEX idx_nodes_depth        ON nodes (depth);

-- Index for parent lookups (children of a node)
CREATE INDEX idx_nodes_parent_id    ON nodes (parent_id);

-- Unique index on node_id
CREATE UNIQUE INDEX idx_nodes_node_id ON nodes (node_id);


-- ============================================================
-- Table 2: chunks
-- Stores actual document content fragments linked to nodes.
-- ============================================================
CREATE TABLE chunks (
    id           SERIAL PRIMARY KEY,
    chunk_id     VARCHAR(50)   NOT NULL,               -- e.g. "chunk_0001"
    node_id      INT           NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    doc_id       VARCHAR(50),                          -- e.g. "doc_001"
    filename     VARCHAR(500),                         -- source filename
    heading      TEXT,                                 -- section heading
    text         TEXT,                                 -- actual document content
    page_approx  INT,                                  -- approximate page number
    summary      TEXT                                  -- chunk-level summary
);

-- Index for fast lookup by chunk_id
CREATE UNIQUE INDEX idx_chunks_chunk_id ON chunks (chunk_id);

-- Index for fetching all chunks belonging to a node
CREATE INDEX idx_chunks_node_id ON chunks (node_id);

-- Index for filtering by source document
CREATE INDEX idx_chunks_filename ON chunks (filename);


-- ============================================================
-- Table 3: node_embeddings
-- Stores vector representations of node summaries.
-- all-MiniLM-L6-v2 produces 384-dimensional vectors.
-- ============================================================
CREATE TABLE node_embeddings (
    id         SERIAL PRIMARY KEY,
    node_id    INT    NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    embedding  vector(384)                             -- sentence-transformers/all-MiniLM-L6-v2
);

-- IVFFlat index for fast approximate cosine similarity search
-- lists=100 is a good default; tune based on row count
CREATE INDEX idx_node_embeddings_vector
    ON node_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for direct node lookup
CREATE UNIQUE INDEX idx_node_embeddings_node_id ON node_embeddings (node_id);


-- ============================================================
-- Done
-- ============================================================