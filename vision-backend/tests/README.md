# Vision Backend Tests

## Quick Start

```bash
# Run all unit tests (fast, ~12s)
pytest tests/unit/ -v

# Run all tests including integration
pytest tests/ -v

# Run unit tests only (skip integration)
pytest tests/ -v -m "not integration"
```

## Structure

| Directory | Purpose |
|-----------|---------|
| `tests/` | Root; smoke tests, shared conftest |
| `tests/unit/` | Fast unit tests (utils, security, use cases, agents) |
| `tests/integration/` | API tests using full app (slower) |

## Phases Implemented

1. **Phase 1** – Test infrastructure (`conftest`, `pytest.ini`, fixtures)
2. **Phase 2** – Pure logic: security, datetime_utils, zone_utils
3. **Phase 3** – Use cases: auth (login, register, get_current_user), camera
4. **Phase 4** – API: auth endpoints with mocked container
5. **Phase 5** – Agents: main_agent (time context, instruction builder, create_agent)

## Test Count

- **Unit tests**: 58 tests (no external deps)
- **Integration tests**: 4 tests (full app lifespan)

## Dependencies

- pytest >= 7.0
- pytest-asyncio
- numpy (for camera helper tests)
