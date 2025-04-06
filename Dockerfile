FROM nvcr.io/nvidia/jax:25.01-py3
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/
WORKDIR /concept

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ENV PATH="/concept/.venv/bin:${PATH}"

# entrypoint is bash so that we get an interactive shell by default
# /root/.bashrc can be used to run shell commands on startup
ENTRYPOINT "/bin/bash"
