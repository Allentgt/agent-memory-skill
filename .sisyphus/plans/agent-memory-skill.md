# Agent Memory Skill - Work Plan

## TL;DR

> **Quick Summary**: Build a Python skill for agent memory using sqlite-memory extension with vectors.space free API for embeddings. Enables semantic search over conversation history and codebase knowledge.
> 
> **Deliverables**:
> - sqlite-memory extension compiled (.dll for Windows)
> - Python skill module with memory add/search/recall functions
> - Example usage and tests
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential build
> **Critical Path**: Build extension → API setup → Python wrapper → Test

---

## Context

### Original Request
Build a local agent memory skill using sqlite-memory with vectors.space free API. Lightweight, entirely local-hosted and free, no subscriptions.

### Interview Summary
**Key Discussions**:
- Use case: Both conversation history + codebase indexing
- Scale: Medium (10K-100K entries)
- Platform: Windows
- Embeddings: vectors.space free API (no signup required)

### Research Findings
- **sqlite-memory**: SQLite extension with hybrid (vector + FTS5) search, markdown-aware chunking
- **vectors.space**: Free embedding API - no signup needed
- **Setup**: Clone → make → Python integration

---

## Work Objectives

### Core Objective
Create a reusable Python skill that provides agent memory capabilities using sqlite-memory + vectors.space

### Concrete Deliverables
1. Compiled sqlite-memory.dll for Windows
2. Python skill module (`agent_memory`)
3. `remember(content, context)` - store memories
4. `recall(query, min_score)` - semantic search
5. Example script and basic tests

### Definition of Done
- [ ] Extension compiles on Windows
- [ ] Python can load extension
- [ ] vectors.space API works
- [ ] Can store and retrieve memories semantically
- [ ] Example shows basic usage

### Must Have
- Fully local storage (SQLite file)
- Free embeddings (vectors.space)
- Python API for agent integration

### Must NOT Have
- No external vector DB servers
- No paid subscriptions
- Not tied to specific LLM provider

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO - this is a new skill
- **Automated tests**: Basic test after implementation
- **Framework**: pytest for Python tests

### QA Policy
Agent-executed QA scenarios after implementation.

---

## Execution Strategy

### Sequential Build (due to C compilation dependency)

```
Wave 1 (Foundation):
├── Task 1: Clone and build sqlite-memory extension for Windows
├── Task 2: Research vectors.space API setup
├── Task 3: Create Python skill module structure
├── Task 4: Implement memory operations (add, search, recall)
└── Task 5: Create example script and test

Critical Path: Task 1 → Task 3 → Task 4 → Task 5
```

---

## TODOs

- [ ] 1. Clone and build sqlite-memory extension for Windows

  **What to do**:
  - Clone sqlite-memory repo with submodules
  - Build using make (check Windows compatibility)
  - Verify .dll file created

  **Must NOT do**:
  - Don't modify source unless needed for Windows build

  **References**:
  - https://github.com/sqliteai/sqlite-memory - README with build instructions
  - Makefile in repo root

  **Acceptance Criteria**:
  - [ ] Extension builds without errors
  - [ ] memory.dll exists in build directory

- [ ] 2. Research vectors.space API setup

  **What to do**:
  - Check vectors.space documentation for free API usage
  - Find API key requirement (if any)
  - Document setup steps

  **References**:
  - https://vectors.space - official site
  - sqlite-memory README shows API usage

  **Acceptance Criteria**:
  - [ ] API endpoint and format documented
  - [ ] No signup requirement verified

- [ ] 3. Create Python skill module structure

  **What to do**:
  - Create `agent_memory/` package directory
  - Create `__init__.py` with skill functions
  - Define `remember(content, context)` function
  - Define `recall(query, min_score)` function

  **References**:
  - sqlite-memory API usage examples from README

  **Acceptance Criteria**:
  - [ ] Package structure created
  - [ ] Functions imported successfully

- [ ] 4. Implement memory operations

  **What to do**:
  - Implement `remember()` - stores content with context
  - Implement `recall()` - semantic search with scoring
  - Handle extension loading
  - Handle API key configuration

  **Acceptance Criteria**:
  - [ ] remember() stores to SQLite
  - [ ] recall() returns relevant results
  - [ ] Works with vectors.space API

- [ ] 5. Create example script and basic test

  **What to do**:
  - Create `example.py` showing usage
  - Create `test_basic.py` with simple test cases
  - Verify full workflow works

  **Acceptance Criteria**:
  - [ ] Example runs successfully
  - [ ] Test passes

---

## Success Criteria

### Verification Commands
```bash
python example.py  # Expected: memories stored and recalled
pytest test_basic.py  # Expected: tests pass
```

### Final Checklist
- [ ] sqlite-memory.dll exists and loads
- [ ] vectors.space API responds
- [ ] remember() stores memories
- [ ] recall() retrieves semantically
- [ ] Example demonstrates workflow