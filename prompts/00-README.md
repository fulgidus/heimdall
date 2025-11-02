# RF Feature Extraction Implementation - Prompt Series

This directory contains a series of incremental prompts for implementing the RF Feature Extraction system in Heimdall SDR.

## Overview

The implementation adds IQ sample generation and feature extraction capabilities to enable ML training on both synthetic and real data.

## Execution Order

Execute these prompts **sequentially** in this exact order:

1. **`01-database-schema.md`** - Create database tables and indexes
2. **`02-feature-extractor-core.md`** - Implement core feature extraction module
3. **`03-iq-generator.md`** - Implement synthetic IQ sample generator
4. **`04-synthetic-pipeline.md`** - Update synthetic data generation pipeline
5. **`05-real-pipeline.md`** - Implement feature extraction for recording sessions
6. **`06-background-jobs.md`** - Add background processing for existing recordings
7. **`07-tests.md`** - Comprehensive test suite

## Verification Between Steps

After each prompt:
1. Run tests if provided
2. Verify the implementation works before proceeding
3. Check that no existing functionality is broken

## Context Needed

Each prompt assumes you have access to:
- The Heimdall codebase at `/home/fulgidus/Documents/Projects/heimdall`
- Docker Compose environment running
- PostgreSQL database accessible
- Python 3.11+ environment

## Key Design Decisions

- **Chunking**: 1000ms IQ → 5 chunks × 200ms for both synthetic and real
- **Aggregation**: Mean + Std + Min + Max = 72 features per receiver
- **Storage**: JSONB array format in PostgreSQL
- **Sample rate**: 200 kHz (2× oversampling for ±50 kHz bandwidth)
- **Parallelization**: Up to 24 workers for synthetic generation
- **IQ retention**: First 100 synthetic samples in MinIO (debug), all real recordings

## Total Feature Count

- 18 base features × 4 aggregations = **72 features per receiver**
- 72 × 7 receivers = **504 total features per sample**
- Storage: ~4 KB per sample vs ~5.6 MB raw IQ = **1400× reduction**

## Expected Outcomes

- Synthetic: 10k samples with features in ~2-3 minutes (24 cores)
- Real: Automatic background extraction for all recordings
- Unified feature format for ML training

## Support

If you encounter issues:
1. Check the relevant prompt's "Verification" section
2. Review the "Common Issues" if provided
3. Ensure previous steps completed successfully
