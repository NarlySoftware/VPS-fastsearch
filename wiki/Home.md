# Welcome to VPS-FastSearch

VPS-FastSearch is a fast hybrid search library combining BM25 full-text search with vector similarity search. It's designed for CPU-only environments like VPS servers where you need instant semantic search without expensive API calls or GPU hardware.

## Quick Links

- **[Getting Started](Getting-Started)** — Install and run your first search in 5 minutes
- **[Use Cases](Use-Cases)** — Real-world examples and integrations
- **[FAQ](FAQ)** — Common questions answered
- **[Comparison](Comparison-with-Alternatives)** — How VPS-FastSearch stacks up against other solutions

## Why VPS-FastSearch?

| Feature | VPS-FastSearch | OpenAI API | Local Sentence-Transformers |
|---------|------------|------------|----------------------------|
| Latency | 4ms | 200-500ms | 800ms+ cold |
| Cost | Free | Per-query | Free |
| Memory | 200-600MB | N/A | 2GB+ |
| Offline | ✅ | ❌ | ✅ |
| CPU-only | ✅ | ✅ | ⚠️ slow |

## Documentation

For detailed technical documentation, see the [docs folder](https://github.com/NarlySoftware/fastsearch/tree/main/docs):

- [Architecture](https://github.com/NarlySoftware/fastsearch/blob/main/docs/ARCHITECTURE.md)
- [CLI Reference](https://github.com/NarlySoftware/fastsearch/blob/main/docs/CLI.md)
- [Python API](https://github.com/NarlySoftware/fastsearch/blob/main/docs/API.md)
- [Configuration](https://github.com/NarlySoftware/fastsearch/blob/main/docs/CONFIGURATION.md)
- [Deployment](https://github.com/NarlySoftware/fastsearch/blob/main/docs/DEPLOYMENT.md)
- [Performance](https://github.com/NarlySoftware/fastsearch/blob/main/docs/PERFORMANCE.md)

## Getting Help

- [GitHub Issues](https://github.com/NarlySoftware/fastsearch/issues) — Bug reports and feature requests
- [Troubleshooting Guide](https://github.com/NarlySoftware/fastsearch/blob/main/docs/TROUBLESHOOTING.md) — Common issues and solutions
