gpu docker compose exec -e CUDA_VISIBLE_DEVICES -e JAX_PLATFORMS -e JAX_CHECK_TRACER_LEAKS=FALSE main uv run "$@"
