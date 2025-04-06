# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run main script: `./yun.sh main.py`
- Run specific function: `./yun.sh -c "from main import specific_function; specific_function()"`
- Run with GPU: Ensure CUDA_VISIBLE_DEVICES is set appropriately
- Docker execution: `docker compose up` or `gpu docker compose up` for GPU usage
- Docker commands: Use `gpu docker compose exec main <command>` to run commands in container

## Code Style Guidelines
- Imports: Group standard library, third-party libraries, then local imports
- Use absolute imports for local modules (from lib import X)
- Use JAX-specific typing annotations (from chex import Array)
- Function naming: snake_case for functions and variables
- Class naming: PascalCase for classes
- Type hints: Use for function parameters and return values 
- Vectorize functions with jax.vmap where possible
- JIT-compile functions with @jax.jit decorator for performance
- Error handling: Use assertions for invariants, proper error messages
- Line length: Keep under 100 characters when possible
- Use PRNGKey objects for randomness and split keys properly